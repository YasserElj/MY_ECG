"""
LeJEPA-ECG Pretraining Script

Self-supervised pretraining using multi-view augmentation and SIGReg regularization.
No masking, no teacher-student, no EMA - just simple invariance + Gaussian regularization.
"""

import argparse
import logging.config
import pprint
import sys
from contextlib import nullcontext
from os import path, makedirs
from time import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import transforms
from data.datasets import DATASETS, MIMIC_IV_ECG, PTB_XL
from data.utils import TensorDataset, DatasetRouter, get_channel_order, load_data_dump
from data.augmentations import MultiViewCollator
from models.lejepa import LeJEPA, SIGReg, compute_lejepa_loss
from utils.monitoring import AverageMeter, get_cpu_count, get_memory_usage
from utils.schedules import cosine_schedule, update_learning_rate_


parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', required=True, help='dataset=path/to/data.npy pairs')
parser.add_argument('--out', default='checkpoints', help='output directory')
parser.add_argument('--config', default='ViTS_mimic', help='config file name')
parser.add_argument('--amp', default='bfloat16', choices=['bfloat16', 'float32'])
args = parser.parse_args()


# Dataset statistics overrides
MIMIC_IV_ECG.mean = [0.000] * 12
MIMIC_IV_ECG.std = [0.155, 0.158, 0.166, 0.133, 0.143, 0.141,
                    0.207, 0.282, 0.284, 0.253, 0.222, 0.197]
PTB_XL.mean = [-0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
               0.000, -0.001, -0.002, -0.001, -0.001, -0.001]
PTB_XL.std = [0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
              0.235, 0.338, 0.335, 0.299, 0.294, 0.242]


def load_config(config_name):
    config_path = path.join(path.dirname(__file__), 'configs', f'{config_name}.yaml')
    if not path.isfile(config_path):
        raise ValueError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class PreprocessECG:
    def __init__(self, mean_std, resample_ratio, channel_order):
        self.mean, self.std = mean_std
        self.resample_ratio = resample_ratio
        self.channel_order = channel_order

    def __call__(self, x):
        transforms.interpolate_NaNs_(x)
        if self.resample_ratio != 1.0:
            channel_size = int(self.resample_ratio * x.shape[0])
            x = transforms.resample(x, channel_size)
        transforms.normalize_(x, mean_std=(self.mean, self.std))
        x = np.clip(x, -5, 5)
        x = x[:, self.channel_order]
        return x


class TransformECG:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        x = transforms.random_crop(x, self.crop_size)
        x = x.transpose()  # channels first
        x = torch.from_numpy(x).float()
        return x


def main():
    makedirs(args.out, exist_ok=True)
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger('app')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    using_cuda = device.type == 'cuda'
    num_cpus = get_cpu_count()
    logger.debug(f'Using {device} accelerator and {num_cpus} CPUs')

    if using_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.debug('TF32 tensor cores enabled')

    if args.amp == 'float32' or not using_cuda:
        logger.debug('Using float32 precision')
        amp_context = nullcontext()
    else:
        logger.debug('Using bfloat16 with AMP')
        amp_context = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    # Load config
    config = load_config(args.config)
    logger.debug(f'Config:\n{pprint.pformat(config, compact=True, width=100)}')

    # Parse data arguments
    dump_files = {}
    for data_arg in args.data:
        dataset_name, dump_file = data_arg.split('=', 1)
        dump_files[dataset_name] = dump_file

    # Load datasets
    datasets = {}
    for dataset_name, weight in config['datasets'].items():
        if dataset_name not in dump_files:
            raise ValueError(f'Missing {dataset_name} in --data argument')
        dump_file = dump_files[dataset_name]
        if not path.isfile(dump_file):
            raise ValueError(f'Dataset not found: {dump_file}')

        logger.debug(f'Loading {dataset_name} from {dump_file}')
        dataset_cls = DATASETS[dataset_name]
        resample_ratio = config['sampling_frequency'] / dataset_cls.sampling_frequency
        channel_order = get_channel_order(dataset_cls.channels, config['channels'])
        mean = np.array([dataset_cls.mean], dtype=np.float16)
        std = np.array([dataset_cls.std], dtype=np.float16)

        dataset = TensorDataset(
            data=load_data_dump(
                dump_file=dump_file,
                transform=PreprocessECG(
                    mean_std=(mean, std),
                    resample_ratio=resample_ratio,
                    channel_order=channel_order),
                processes=num_cpus),
            transform=TransformECG(crop_size=config['channel_size'])
        )
        datasets[dataset_name] = (dataset, weight)

    logger.debug(f'{get_memory_usage() / 1024**3:.2f}GB memory used')

    # Create multi-view collator
    collate_fn = MultiViewCollator(
        num_views=config['num_views'],
        crop_size=config['channel_size'],
        crop_scale=tuple(config['crop_scale']),
        amplitude_scale=tuple(config['amplitude_scale']),
        noise_std=config['noise_std']
    )

    train_loader = DataLoader(
        dataset=DatasetRouter(datasets.values()),
        batch_size=config['batch_size'],
        pin_memory=using_cuda,
        collate_fn=collate_fn,
        num_workers=2
    )

    def map_to_device(iterator):
        for batch in iterator:
            yield batch.to(device, non_blocking=True)

    train_iterator = iter(train_loader)
    train_iterator = map_to_device(train_iterator)

    # Create model
    model = LeJEPA(
        dim=config['dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        num_channels=len(config['channels']),
        channel_size=config['channel_size'],
        patch_size=config['patch_size'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        bias=config['bias'],
        dropout=config['dropout'],
        attn_dropout=config['attn_dropout'],
        norm_eps=config['norm_eps'],
        layer_scale_eps=config['layer_scale_eps'],
        num_registers=config['num_registers'],
        proj_hidden_dim=config['proj_hidden_dim'],
        proj_dim=config['proj_dim'],
        use_sdp_kernel=using_cuda
    ).to(device)

    # Create SIGReg loss
    sigreg = SIGReg(num_slices=config['num_slices']).to(device)

    # Create optimizer
    optimizer = model.get_optimizer(
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        fused=using_cuda
    )

    # Learning rate schedule
    lr_schedule = cosine_schedule(
        total_steps=config['steps'],
        start_value=config['learning_rate'],
        final_value=config['final_learning_rate'],
        warmup_steps=config['learning_rate_warmup_steps'],
        warmup_start_value=1e-6
    )

    # Training
    lamb = config['lambda']
    step_time = AverageMeter()
    train_loss = AverageMeter()
    inv_loss_meter = AverageMeter()
    sigreg_loss_meter = AverageMeter()
    best_loss = None

    logger.info(f'Starting LeJEPA pretraining for {config["steps"]} steps')
    logger.info(f'λ={lamb}, num_views={config["num_views"]}, proj_dim={config["proj_dim"]}')

    for step in range(config['steps']):
        step_start = time()

        # Update learning rate
        update_learning_rate_(optimizer, next(lr_schedule))

        # Forward pass
        batch_loss = 0.
        batch_inv = 0.
        batch_sigreg = 0.

        for _ in range(config['gradient_accumulation_steps']):
            views = next(train_iterator)  # (B, V, C, T)

            with amp_context:
                emb, proj = model(views)
                loss, inv_loss, sigreg_loss = compute_lejepa_loss(proj, sigreg, lamb)
                loss = loss / config['gradient_accumulation_steps']

            loss.backward()
            batch_loss += loss.item()
            batch_inv += inv_loss.item() / config['gradient_accumulation_steps']
            batch_sigreg += sigreg_loss.item() / config['gradient_accumulation_steps']

        # Gradient clipping
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update meters
        train_loss.update(batch_loss)
        inv_loss_meter.update(batch_inv)
        sigreg_loss_meter.update(batch_sigreg)
        step_time.update(time() - step_start)

        # Logging
        if (step + 1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'[{step + 1:06d}] '
                f'step_time={step_time.value:.4f} '
                f'loss={train_loss.value:.4f} '
                f'inv={inv_loss_meter.value:.4f} '
                f'sigreg={sigreg_loss_meter.value:.4f} '
                f'lr={lr:.2e}'
            )
            step_time = AverageMeter()
            train_loss = AverageMeter()
            inv_loss_meter = AverageMeter()
            sigreg_loss_meter = AverageMeter()

        # Checkpoint
        if (step + 1) % config['checkpoint_interval'] == 0:
            ckpt_path = path.join(args.out, f'chkpt_{step + 1}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'step': step + 1,
            }, ckpt_path)
            logger.info(f'Saved {ckpt_path}')

        # Best checkpoint
        if best_loss is None or batch_loss < best_loss:
            best_loss = batch_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'step': step + 1,
                'loss': best_loss,
            }, path.join(args.out, 'best_ckpt.pt'))
            if (step + 1) % 100 == 0:
                logger.info(f'[BEST] step {step + 1} | loss={best_loss:.4f}')

    logger.info('Training complete!')


if __name__ == '__main__':
    main()

