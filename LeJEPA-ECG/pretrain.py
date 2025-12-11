"""
LeJEPA-ECG Pretraining Script

Self-supervised pretraining using multi-view augmentation and SIGReg regularization.
No masking, no teacher-student, no EMA - just simple invariance + Gaussian regularization.

Supports both single-GPU and multi-GPU (DDP) training:
  Single GPU:  python pretrain.py --config ViTS_mimic ...
  Multi-GPU:   torchrun --nproc_per_node=2 pretrain.py --config ViTS_mimic ...
"""

import argparse
import logging.config
import os
import pprint
import random
from contextlib import nullcontext
from datetime import datetime
from os import path, makedirs
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """
    Setup distributed training if running with torchrun.
    Returns (local_rank, world_size, is_main_process).
    """
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if local_rank >= 0:
        # Running with torchrun
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        is_main = local_rank == 0
    else:
        # Single GPU mode
        local_rank = 0
        world_size = 1
        is_main = True
    
    return local_rank, world_size, is_main


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
parser.add_argument('--entity', default='AtlasVision_CC', help='wandb team/entity name')
parser.add_argument('--run-name', default=None, help='wandb run name')
parser.add_argument('--resume', default=None, help='path to checkpoint to resume from')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
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


def get_gpu_stats():
    """Get GPU memory and utilization stats."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    
    return {
        'system/gpu_memory_allocated_gb': allocated,
        'system/gpu_memory_reserved_gb': reserved,
        'system/gpu_peak_memory_gb': peak,
        'system/gpu_utilization_pct': (allocated / reserved * 100) if reserved > 0 else 0,
    }


def get_gradient_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def create_pca_plot(embeddings, n_samples=1000):
    """Create PCA visualization of embeddings."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    emb = embeddings.detach().cpu().numpy()
    if len(emb) > n_samples:
        idx = np.random.choice(len(emb), n_samples, replace=False)
        emb = emb[idx]
    
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, s=10, c=np.arange(len(emb_2d)), cmap='viridis')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Embedding PCA')
    plt.colorbar(scatter, ax=ax, label='Sample index')
    plt.tight_layout()
    return fig


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
    # Setup distributed training (will be single GPU if not using torchrun)
    local_rank, world_size, is_main = setup_distributed()
    distributed = world_size > 1
    
    makedirs(args.out, exist_ok=True)
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger('app')
    
    # Only log from main process
    if not is_main:
        logger.setLevel(logging.WARNING)

    # Set random seed for reproducibility (offset by rank for variety)
    set_seed(args.seed + local_rank)
    if is_main:
        logger.info(f'Random seed set to {args.seed}')

    # Set device based on local rank
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        using_cuda = True
    else:
        device = torch.device('cpu')
        using_cuda = False
    
    num_cpus = get_cpu_count()
    if is_main:
        if distributed:
            logger.info(f'Distributed training: {world_size} GPUs')
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

    # Wandb config
    use_wandb = args.wandb and WANDB_AVAILABLE
    wandb_log_interval = config.get('wandb_log_interval', 10)
    wandb_viz_interval = config.get('wandb_viz_interval', 1000)
    wandb_project = config.get('wandb_project', 'LeJEPA-ECG')
    
    if args.wandb and not WANDB_AVAILABLE:
        logger.warning('wandb not installed, skipping wandb logging')
    
    if use_wandb:
        run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            entity=args.entity,
            project=wandb_project,
            name=run_name,
            config={**config, 'seed': args.seed},
            tags=['lejepa', 'ecg', 'pretraining']
        )
        logger.info(f'Wandb initialized: {args.entity}/{wandb_project}/{run_name}')

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

    # Create dataset
    train_dataset = DatasetRouter(datasets.values())
    
    # Use DistributedSampler for multi-GPU training
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            drop_last=True
        )
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        pin_memory=using_cuda,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=True
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
    
    # Wrap model in DDP for distributed training
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Log model info
    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    if is_main:
        logger.info(f'Model parameters: {num_params / 1e6:.2f}M')
        if distributed:
            effective_batch = config['batch_size'] * world_size * config['gradient_accumulation_steps']
            logger.info(f'Effective batch size: {effective_batch} (batch={config["batch_size"]} x gpus={world_size} x accum={config["gradient_accumulation_steps"]})')
    if use_wandb:
        wandb.config.update({'num_params': num_params, 'world_size': world_size})

    # Create SIGReg loss
    sigreg = SIGReg(num_slices=config['num_slices']).to(device)

    # Create optimizer (use model_without_ddp to get optimizer from LeJEPA class)
    optimizer = model_without_ddp.get_optimizer(
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        fused=using_cuda
    )

    # Resume from checkpoint
    start_step = 0
    best_loss = None
    total_samples = 0
    
    if args.resume:
        if path.isfile(args.resume):
            if is_main:
                logger.info(f'Resuming from checkpoint: {args.resume}')
            ckpt = torch.load(args.resume, map_location=device)
            model_without_ddp.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_step = ckpt.get('step', 0)
            best_loss = ckpt.get('loss', None)
            total_samples = ckpt.get('total_samples', 0)
            if is_main:
                logger.info(f'Resumed from step {start_step}, best_loss={best_loss}')
        else:
            if is_main:
                logger.warning(f'Checkpoint not found: {args.resume}, starting from scratch')

    # Learning rate schedule (create full schedule, will skip to start_step)
    lr_schedule = cosine_schedule(
        total_steps=config['steps'],
        start_value=config['learning_rate'],
        final_value=config['final_learning_rate'],
        warmup_steps=config['learning_rate_warmup_steps'],
        warmup_start_value=1e-6
    )
    
    # Skip LR schedule to start_step
    for _ in range(start_step):
        next(lr_schedule)

    # Training
    lamb = config['lambda']
    step_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = AverageMeter()
    inv_loss_meter = AverageMeter()
    sigreg_loss_meter = AverageMeter()

    remaining_steps = config['steps'] - start_step
    if is_main:
        logger.info(f'Starting LeJEPA pretraining for {remaining_steps} steps (from {start_step} to {config["steps"]})')
        logger.info(f'λ={lamb}, num_views={config["num_views"]}, proj_dim={config["proj_dim"]}')

    for step in range(start_step, config['steps']):
        step_start = time()

        # Update learning rate
        lr = next(lr_schedule)
        update_learning_rate_(optimizer, lr)

        # Forward pass
        batch_loss = 0.
        batch_inv = 0.
        batch_sigreg = 0.
        last_emb = None
        last_proj = None

        for _ in range(config['gradient_accumulation_steps']):
            data_start = time()
            views = next(train_iterator)  # (B, V, C, T)
            data_time.update(time() - data_start)
            
            batch_size = views.size(0)
            # Total samples across all GPUs
            total_samples += batch_size * world_size

            with amp_context:
                emb, proj = model(views)
                loss, inv_loss, sigreg_loss = compute_lejepa_loss(proj, sigreg, lamb)
                loss = loss / config['gradient_accumulation_steps']
            
            last_emb = emb.detach()
            last_proj = proj.detach()

            loss.backward()
            batch_loss += loss.item()
            batch_inv += inv_loss.item() / config['gradient_accumulation_steps']
            batch_sigreg += sigreg_loss.item() / config['gradient_accumulation_steps']

        # Get gradient norm before clipping
        grad_norm = get_gradient_norm(model)

        # Gradient clipping
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update meters
        step_elapsed = time() - step_start
        train_loss.update(batch_loss)
        inv_loss_meter.update(batch_inv)
        sigreg_loss_meter.update(batch_sigreg)
        step_time.update(step_elapsed)
        
        # Throughput: samples per second across all GPUs
        throughput = batch_size * config['gradient_accumulation_steps'] * world_size / step_elapsed

        # Wandb logging (every step or interval, main process only)
        if use_wandb and is_main and (step + 1) % wandb_log_interval == 0:
            log_dict = {
                'train/loss': batch_loss,
                'train/inv_loss': batch_inv,
                'train/sigreg_loss': batch_sigreg,
                'train/lr': lr,
                'train/step_time': step_elapsed,
                'train/throughput': throughput,
                'train/total_samples': total_samples,
                'gradients/norm': grad_norm,
            }
            
            # Embedding stats
            if last_proj is not None:
                proj_flat = last_proj.reshape(-1, last_proj.size(-1))
                log_dict.update({
                    'embeddings/proj_mean': proj_flat.mean().item(),
                    'embeddings/proj_std': proj_flat.std().item(),
                    'embeddings/proj_norm': proj_flat.norm(dim=-1).mean().item(),
                })
            
            # GPU stats - log every time for visibility
            log_dict.update(get_gpu_stats())
            
            wandb.log(log_dict, step=step + 1)

        # PCA visualization (main process only)
        if use_wandb and is_main and (step + 1) % wandb_viz_interval == 0 and last_emb is not None:
            try:
                fig = create_pca_plot(last_emb)
                wandb.log({'embeddings/pca': wandb.Image(fig)}, step=step + 1)
                plt.close(fig)
            except Exception as e:
                logger.warning(f'Failed to create PCA plot: {e}')

        # Console logging (main process only)
        if is_main and (step + 1) % 100 == 0:
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if using_cuda else 0
            logger.info(
                f'[{step + 1:06d}] '
                f'time={step_time.value:.3f}s '
                f'loss={train_loss.value:.4f} '
                f'inv={inv_loss_meter.value:.4f} '
                f'sigreg={sigreg_loss_meter.value:.2f} '
                f'lr={lr:.2e} '
                f'gpu={gpu_mem:.1f}GB (peak)'
            )
            step_time = AverageMeter()
            data_time = AverageMeter()
            train_loss = AverageMeter()
            inv_loss_meter = AverageMeter()
            sigreg_loss_meter = AverageMeter()

        # Checkpoint (with barrier to sync all GPUs, only main process saves)
        if (step + 1) % config['checkpoint_interval'] == 0:
            if distributed:
                dist.barrier()  # Wait for all ranks before saving
            if is_main:
                ckpt_path = path.join(args.out, f'chkpt_{step + 1}.pt')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'step': step + 1,
                    'loss': train_loss.value,
                    'total_samples': total_samples,
                    'seed': args.seed,
                }, ckpt_path)
                logger.info(f'Saved {ckpt_path}')
                
                if use_wandb:
                    wandb.save(ckpt_path)

        # Best checkpoint (main process only, check every 100 steps using smoothed loss)
        if is_main and (step + 1) % 100 == 0:
            avg_loss = train_loss.value
            if avg_loss is not None and (best_loss is None or avg_loss < best_loss):
                best_loss = avg_loss
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'step': step + 1,
                    'loss': best_loss,
                    'total_samples': total_samples,
                    'seed': args.seed,
                }, path.join(args.out, 'best_ckpt.pt'))
                logger.info(f'[BEST] step {step + 1} | loss={best_loss:.4f}')

    if is_main:
        logger.info('Training complete!')
    
    if use_wandb and is_main:
        wandb.finish()
    
    # Cleanup distributed
    cleanup_distributed()


if __name__ == '__main__':
    # Import matplotlib here to avoid issues if not installed
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
