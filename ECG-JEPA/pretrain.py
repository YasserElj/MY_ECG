import argparse
import dataclasses
import logging.config
import pprint
from contextlib import nullcontext
from os import path, makedirs
from time import time


import numpy as np
import torch
from torch.utils.data import DataLoader

import configs
from data import transforms, utils as datautils
from data.datasets import (
    DATASETS,
    CODE15,
    StPetersburg,
    PTB_XL
)
from data.masks import MaskCollator, IJEPAMaskCollator, IJEPAMaskCollatorFlat
from data.utils import (
    TensorDataset,
    VariableTensorDataset,
    DatasetRouter
)
from models import JEPA
from models.IJEPA import IJEPA
from utils.monitoring import (
    AverageMeter,
    get_cpu_count,
    get_memory_usage
)
from utils.schedules import (
    linear_schedule,
    cosine_schedule,
    update_weight_decay_,
    update_learning_rate_
)

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', required=True, help='list of dataset=path/to/data.npy pairs')
parser.add_argument('--out', default='pretrain', help='output directory')
parser.add_argument('--config', default='ViTS_mimic', help='path to config file or config name')
parser.add_argument('--chkpt', help='resume training from model checkpoint')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'], help='automated mixed precision')
parser.add_argument('--compile', action='store_true', help='compile model')
args = parser.parse_args()

# NOTE: we update means and standard deviations of some datasets
#  because we use their preprocessed version instead of the original.
#  The preprocessed versions have had their baseline wander removed.
#  This was essential to maintain training stability.
CODE15.mean = [0.000] * len(CODE15.channels)
CODE15.std = [0.488, 0.450, 0.437, 0.416, 0.405, 0.370,
              0.548, 0.639, 0.719, 0.695, 0.676, 0.639]
StPetersburg.mean = [0.000] * len(StPetersburg.channels)
StPetersburg.std = [0.132, 0.370, 0.353, 0.215, 0.191, 0.356,
                    0.234, 0.320, 0.328, 0.290, 0.317, 0.337]
# NOTE: we compute mean and standard deviation of ptb-xl over the train folds (1-8).
#  We only use these folds during pre-training.
PTB_XL.mean = [-0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
               0.000, -0.001, -0.002, -0.001, -0.001, -0.001]
PTB_XL.std = [0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
              0.235, 0.338, 0.335, 0.299, 0.294, 0.242]


def main():
    makedirs(args.out, exist_ok=True)
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger('app')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    using_cuda = device.type == 'cuda'
    num_cpus = get_cpu_count()
    logger.debug(f'using {device} accelerator and {num_cpus} CPUs')

    if using_cuda:
        logger.debug('TF32 tensor cores are enabled')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.amp == 'float32' or not using_cuda:
        logger.debug('using float32 precision')
        auto_mixed_precision = nullcontext()
    elif args.amp == 'bfloat16':
        logger.debug('using bfloat16 with AMP')
        auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        raise ValueError('Failed to choose floating-point format.')

    if args.chkpt:
        logger.debug(f'resuming from checkpoint {args.chkpt}')
        chkpt = torch.load(args.chkpt, map_location=device)
        config = configs.pretrain.Config(**chkpt['config'])
    else:
        if not path.isfile(args.config):
            config_file = path.join(path.dirname(configs.pretrain.__file__), f'{args.config}.yaml')
            if not path.isfile(config_file):
                raise ValueError(f'Failed to read configuration file {args.config}')
            args.config = config_file
        config_dict = configs.load_config_file(args.config)
        config = configs.pretrain.Config(**config_dict)
        logger.debug(f'loading configuration file from {args.config}:\n'
                     f'{pprint.pformat(config_dict, compact=True, sort_dicts=False, width=120)}')
        chkpt = None

    dump_files = {}
    for data_arg in args.data:
        dataset_name, *maybe_dump_file = data_arg.split('=', 1)
        if not maybe_dump_file:
            raise ValueError('Dataset pair must have following format: dataset=path/to/data.npy')
        dump_file, = maybe_dump_file
        dump_files[dataset_name] = dump_file

    for dataset_name in config.datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f'Unknown dataset {dataset_name}. '
                             f'Available datasets are {list(DATASETS)}')
        if dataset_name not in dump_files:
            raise ValueError(f'Missing {dataset_name} dataset in `--data` argument')
        dump_file = dump_files[dataset_name]
        if not path.isfile(dump_file):
            raise ValueError(f'Dataset does not exist {dump_file}')
        _, ext = path.splitext(dump_file)
        if ext not in ('.npy', '.npz'):
            raise ValueError(f'Unsupported dataset format: {dump_file}')

    datasets = {}
    for dataset_name, weight in config.datasets.items():
        dump_file = dump_files[dataset_name]
        logger.debug(f'loading {dataset_name} from {dump_file}')
        dataset_cls = DATASETS[dataset_name]
        resample_ratio = config.sampling_frequency / dataset_cls.sampling_frequency
        channel_order = datautils.get_channel_order(dataset_cls.channels, config.channels)
        mean = np.array([dataset_cls.mean], dtype=np.float16)
        std = np.array([dataset_cls.std], dtype=np.float16)
        _, ext = path.splitext(dump_file)
        if ext == '.npy':
            dataset = TensorDataset(
                data=datautils.load_data_dump(
                    dump_file=dump_file,
                    transform=PreprocessECG(
                        mean_std=(mean, std),
                        resample_ratio=resample_ratio,
                        channel_order=channel_order),
                    processes=num_cpus),
                transform=TransformECG(crop_size=config.channel_size))
        elif ext == '.npz':
            dataset = VariableTensorDataset(
                *load_variable_data_dump(
                    dump_file=dump_file,
                    min_channel_size=config.channel_size,
                    transform=PreprocessECG(
                        mean_std=(mean, std),
                        resample_ratio=resample_ratio,
                        channel_order=channel_order),
                    processes=num_cpus),
                transform=TransformECG(crop_size=config.channel_size))
        else:
            raise ValueError(f'Unsupported dataset format: {dump_file}')
        datasets[dataset_name] = (dataset, weight)

    logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

    # Determine masking strategy and model type
    masking_strategy = getattr(config, 'masking_strategy', 'ecg-jepa')
    use_ijepa_model = masking_strategy == 'i-jepa'

    if masking_strategy == 'i-jepa':
        logger.info('Using I-JEPA context-to-target masking strategy with IJEPA model.')
        collate_fn = IJEPAMaskCollator(
            patch_size=config.patch_size,
            context_scale=getattr(config, 'context_scale', (0.85, 0.95)),
            pred_scale=getattr(config, 'pred_scale', (0.15, 0.20)),
            num_pred_blocks=getattr(config, 'n_pred_blocks', 4),
            min_keep=getattr(config, 'min_keep', 10)
        )
    elif masking_strategy == 'i-jepa-flat':
        logger.info('Using I-JEPA masking with original JEPA model (flattened).')
        use_ijepa_model = False
        collate_fn = IJEPAMaskCollatorFlat(
            patch_size=config.patch_size,
            context_scale=getattr(config, 'context_scale', (0.85, 0.95)),
            pred_scale=getattr(config, 'pred_scale', (0.15, 0.20)),
            num_pred_blocks=getattr(config, 'n_pred_blocks', 4),
            min_keep=getattr(config, 'min_keep', 10)
        )
    else:
        logger.info('Using ECG-JEPA mask-and-complement masking strategy.')
        use_ijepa_model = False
        collate_fn = MaskCollator(
            patch_size=config.patch_size,
            min_block_size=config.min_block_size,
            min_keep_ratio=config.min_keep_ratio,
            max_keep_ratio=config.max_keep_ratio
        )

    train_loader = DataLoader(
        dataset=DatasetRouter(datasets.values()),
        batch_size=config.batch_size,
        pin_memory=using_cuda,
        collate_fn=collate_fn,
        num_workers=2)

    def map_to_device(data_iterator, device=None):
        for batch in data_iterator:
            result = []
            for item in batch:
                if isinstance(item, list):
                    result.append([t.to(device, non_blocking=using_cuda) for t in item])
                else:
                    result.append(item.to(device, non_blocking=using_cuda))
            yield tuple(result)

    def prefetch_batch(data_iterator):
        prefetched_batch = next(data_iterator)
        for next_batch in data_iterator:
            yield prefetched_batch
            prefetched_batch = next_batch
        yield prefetched_batch

    train_iterator = iter(train_loader)
    train_iterator = map_to_device(train_iterator, device=device)
    train_iterator = prefetch_batch(train_iterator)

    # setup hyperparameter schedules
    if chkpt is not None:
        step = chkpt['step']
    else:
        step = 0

    momentum_schedule = linear_schedule(
        total_steps=config.steps,
        start_value=config.encoder_momentum,
        final_value=config.final_encoder_momentum,
        step=step)
    lr_schedule = cosine_schedule(
        total_steps=config.steps,
        start_value=config.learning_rate,
        final_value=config.final_learning_rate,
        warmup_steps=config.learning_rate_warmup_steps,
        warmup_start_value=1e-6,
        step=step)
    wd_schedule = cosine_schedule(
        total_steps=config.steps,
        start_value=config.weight_decay,
        final_value=config.final_weight_decay,
        step=step)

    # setup model
    if use_ijepa_model:
        logger.info('Initializing IJEPA model (multi-block predictor).')
        model = original_model = IJEPA(
            config=config,
            momentum_schedule=momentum_schedule,
            use_sdp_kernel=using_cuda
        ).to(device)
    else:
        logger.info('Initializing original JEPA model.')
        model = original_model = JEPA(
            config=config,
            momentum_schedule=momentum_schedule,
            use_sdp_kernel=using_cuda
        ).to(device)
    optimizer = model.get_optimizer(fused=using_cuda)

    if chkpt is not None:
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])

    if args.compile:
        model = torch.compile(model)

    step_time = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    train_loss = AverageMeter()

    best_loss = None

    for step in range(config.steps):
        step_start = time()
        update_learning_rate_(optimizer, next(lr_schedule))
        update_weight_decay_(optimizer, next(wd_schedule))
        batch_loss = 0.
        for _ in range(config.gradient_accumulation_steps):
            data_start = time()
            x, mask_encoder, mask_predictor = next(train_iterator)
            data_time.update(time() - data_start)

            if use_ijepa_model:
                num_patches = x.size(-1) // config.patch_size
                if isinstance(mask_encoder, list):
                    kr = mask_encoder[0].size(1) / num_patches
                else:
                    kr = mask_encoder.size(1) / num_patches
            else:
                if isinstance(mask_encoder, list):
                    mask_encoder = mask_encoder[0]
                if isinstance(mask_predictor, list):
                    mask_predictor = torch.cat(mask_predictor, dim=1)
                elif mask_predictor.dim() == 3:
                    B, N, L = mask_predictor.shape
                    mask_predictor = mask_predictor.reshape(B, N * L)
                kr = mask_encoder.float().mean().item()

            if (step + 1) % 100 == 0:
                logging.getLogger('app').info(f"[mask] keep_ratio≈{kr:.3f}")
            forward_start = time()
            with auto_mixed_precision:
                loss = model(x, mask_encoder, mask_predictor)
                loss = loss / config.gradient_accumulation_steps
            loss.backward()
            forward_time.update(time() - forward_start)
            batch_loss += loss.item()

        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        train_loss.update(batch_loss)
        optimizer.zero_grad(set_to_none=True)
        step_end = time()
        step_time.update(step_end - step_start)

        if (step + 1) % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            logging.getLogger('app').info(f"[sched] step={step+1} lr={lr:.3e} wd={wd:.3g}")

        if (step + 1) % 100 == 0:
            logger.info(f'[{step + 1:06d}] '
                        f'step_time {step_time.value:.4f} '
                        f'data_time {data_time.value:.4f} '
                        f'forward_time {forward_time.value:.4f} '
                        f'train_loss {train_loss.value:.4f}')
            step_time = AverageMeter()
            data_time = AverageMeter()
            forward_time = AverageMeter()
            train_loss = AverageMeter()

        if (step + 1) % 200 == 0:
            xm, xs = x.mean().item(), x.std().item()
            logging.getLogger('app').debug(f"[data] x mean={xm:.3f} std={xs:.3f}")

        if (step + 1) % config.checkpoint_interval == 0:
            torch.save({
                'model': original_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': dataclasses.asdict(config),
                'step': step + 1,
            }, path.join(args.out, f'chkpt_{step + 1}.pt'))
            logger.info(f'saved chkpt_{step + 1}.pt')

        loss_val = batch_loss
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            torch.save({
                "model": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": dataclasses.asdict(config),
                "step": step + 1,
                "loss": best_loss,
            }, path.join(args.out, "best_ckpt.pt"))
            logger.info(f"[BEST] step {step+1} | loss={loss_val:.4f} | saved best_ckpt.pt")


def load_variable_data_dump(dump_file, min_channel_size, transform=None, processes=None):
    data = datautils.load_variable_data_dump(dump_file, transform=transform, processes=processes)
    data = [x for x in data if len(x) >= min_channel_size]
    sizes = np.array([len(x) for x in data])
    starts = np.concatenate([np.array([0]), np.cumsum(sizes[:-1])])
    data = np.concatenate(data)
    return data, starts, sizes


class PreprocessECG:
    def __init__(self, *, mean_std, resample_ratio, channel_order):
        self.mean, self.std = mean_std
        self.resample_ratio = resample_ratio
        self.channel_order = channel_order

    def __call__(self, x):
        transforms.interpolate_NaNs_(x)
        if self.resample_ratio != 1.0:
            channel_size, num_channels = x.shape
            channel_size = int(self.resample_ratio * channel_size)
            x = transforms.resample(x, channel_size)
        transforms.normalize_(x, mean_std=(self.mean, self.std))
        x.clip(-5, 5, out=x)
        x = x[:, self.channel_order]
        return x


class TransformECG:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        x = transforms.random_crop(x, self.crop_size)
        x = x.transpose()
        x = torch.from_numpy(x).float()
        return x


if __name__ == '__main__':
    main()
