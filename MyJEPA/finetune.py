"""
MyJEPA Finetuning Script

Fine-tune a pre-trained MyJEPA encoder on PTB-XL classification tasks.
Supports wandb logging for comprehensive experiment tracking.
"""

import argparse
import copy
import logging.config
import pprint
import random
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from os import path, makedirs
from time import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from torch.nn import functional as F
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import configs
from data import transforms
from data.datasets import PTB_XL
from data.utils import TensorDataset, get_channel_order, load_data_dump
from models.vit import VisionTransformer
from models.vit_classifier import ViTClassifier
from utils.monitoring import AverageMeter, get_memory_usage, get_cpu_count
from utils.schedules import update_learning_rate_, cosine_schedule

# PTB-XL Tasks
TASKS = (
    'all',
    'diagnostic',
    'subdiagnostic',
    'superdiagnostic',
    'form',
    'rhythm',
    'ST-MEM',
)
FOLDS = tuple(range(1, 11))

# ViT config keys to extract from MyJEPA checkpoint
VIT_CONFIG_KEYS = [
    'sampling_frequency', 'channels', 'channel_size', 'patch_size',
    'dim', 'depth', 'num_heads', 'mlp_ratio', 'qkv_bias', 'bias',
    'dropout', 'attn_dropout', 'num_registers', 'norm_eps', 'layer_scale_eps'
]

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to PTB-XL data directory')
parser.add_argument('--encoder', required=True, help='path to MyJEPA checkpoint')
parser.add_argument('--out', default='eval', help='output directory')
parser.add_argument('--config', default='linear', help='path to config file or config name')
parser.add_argument('--dump', help='path to dump file (.npy) with raw ECG signals')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'])
parser.add_argument('--task', choices=TASKS, default='all', help='task type')
parser.add_argument('--val-fold', choices=FOLDS, type=int, default=9, help='validation fold')
parser.add_argument('--test-fold', choices=FOLDS, type=int, default=10, help='test fold')
parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
parser.add_argument('--entity', default='AtlasVision_CC', help='wandb team/entity name')
parser.add_argument('--run-name', default=None, help='wandb run name')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
args = parser.parse_args()


@dataclass
class EncoderConfig:
    """ViT encoder configuration extracted from MyJEPA checkpoint."""
    sampling_frequency: int
    channels: tuple
    channel_size: int
    patch_size: int
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    bias: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    num_registers: int = 0
    norm_eps: float = 1e-6
    layer_scale_eps: float = 0.0

    @property
    def num_channels(self):
        return len(self.channels)


def get_gpu_stats():
    """Get GPU memory stats."""
    if not torch.cuda.is_available():
        return {}
    return {
        'system/gpu_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
    }


def create_embedding_plot(
    model, 
    dataloader, 
    device, 
    class_names, 
    title="Embeddings",
    num_samples=1000, 
    perplexity=30,
    single_label=False
):
    """
    Create t-SNE embedding plot colored by class labels.
    
    Args:
        model: ViTClassifier model
        dataloader: DataLoader with (x, y) batches
        device: torch device
        class_names: list of class names
        title: plot title
        num_samples: max samples to visualize
        perplexity: t-SNE perplexity
        single_label: if True, use argmax for labels; else use first positive class
    
    Returns:
        matplotlib figure
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            if len(embeddings) * x_batch.shape[0] >= num_samples:
                break
            x_batch = x_batch.to(device)
            
            # Handle cropped inputs (B, num_crops, C, L) -> (B*num_crops, C, L)
            if x_batch.dim() == 4:
                batch_size, num_crops = x_batch.shape[:2]
                x_batch = x_batch.reshape(-1, x_batch.shape[2], x_batch.shape[3])
                # Get encoder embeddings (before classifier head)
                emb = model.encoder(x_batch)  # (B*num_crops, N, dim)
                emb = emb.mean(dim=1)  # Global average pool -> (B*num_crops, dim)
                # Average across crops
                emb = emb.reshape(batch_size, num_crops, -1).mean(dim=1)  # (B, dim)
            else:
                # Get encoder embeddings (before classifier head)
                emb = model.encoder(x_batch)  # (B, N, dim)
                emb = emb.mean(dim=1)  # Global average pool -> (B, dim)
            
            embeddings.append(emb.cpu())
            labels.append(y_batch.cpu())
    
    embeddings = torch.cat(embeddings)[:num_samples].numpy()
    labels_tensor = torch.cat(labels)[:num_samples]
    
    # Convert multi-label to single label for coloring
    if single_label:
        # Single-label: use argmax
        label_indices = labels_tensor.argmax(dim=1).numpy()
    else:
        # Multi-label: use first positive class (or -1 if none)
        label_indices = []
        for row in labels_tensor:
            positive_indices = torch.where(row > 0.5)[0]
            if len(positive_indices) > 0:
                label_indices.append(positive_indices[0].item())
            else:
                label_indices.append(-1)
        label_indices = np.array(label_indices)
    
    # t-SNE projection
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings) - 1), 
                random_state=42, n_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)
    
    # Create plot with nice styling
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique labels and create colormap
    unique_labels = np.unique(label_indices)
    num_classes = len(unique_labels)
    
    # Use a good colormap
    if num_classes <= 10:
        cmap = plt.cm.tab10
    elif num_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis
    
    # Plot each class
    for i, label_idx in enumerate(unique_labels):
        mask = label_indices == label_idx
        if label_idx == -1:
            name = "Unknown"
            color = 'gray'
        else:
            name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
            color = cmap(i / max(num_classes - 1, 1))
        
        ax.scatter(
            emb_2d[mask, 0], 
            emb_2d[mask, 1], 
            c=[color], 
            label=f"{name} ({mask.sum()})",
            alpha=0.6, 
            s=15,
            edgecolors='none'
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    
    # Legend outside plot
    ax.legend(
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9
    )
    
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    model.train()
    return fig


def main():
    makedirs(args.out, exist_ok=True)
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger('app')

    set_seed(args.seed)
    logger.info(f'Random seed set to {args.seed}')

    dump_file = args.dump or f'{args.data_dir}.npy'
    if not path.isfile(dump_file):
        raise ValueError(f'Failed to find .npy data file. Attempted location: {dump_file}. '
                         f'Use `--dump` to specify location.')

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

    # Load eval config
    if not path.isfile(args.config):
        config_file = path.join(path.dirname(configs.eval.__file__), f'{args.config}.yaml')
        if not path.isfile(config_file):
            raise ValueError(f'Failed to read configuration file {args.config}')
        args.config = config_file

    eval_config_dict = configs.load_config_file(args.config)
    logger.debug(f'loading configuration file from {args.config}\n'
                 f'{pprint.pformat(eval_config_dict, compact=True, sort_dicts=False, width=120)}')

    # Load MyJEPA checkpoint
    logger.debug(f'loading encoder checkpoint from {args.encoder}')
    chkpt = torch.load(args.encoder, map_location='cpu')
    
    # Extract ViT config (filter MyJEPA-specific keys)
    full_config = chkpt['config']
    encoder_config_dict = {k: full_config[k] for k in VIT_CONFIG_KEYS if k in full_config}
    encoder_config_dict['channels'] = tuple(encoder_config_dict['channels'])
    encoder_config = EncoderConfig(**encoder_config_dict)
    
    # Extract encoder weights from MyJEPA model
    # MyJEPA saves encoder as 'encoder.*'
    model_state_dict = {
        'encoder.' + k.removeprefix('encoder.'): v
        for k, v in chkpt['model'].items()
        if k.startswith('encoder.')
    }
    
    logger.debug(f'extracted {len(model_state_dict)} encoder parameters from checkpoint')

    # Task setup
    ptb_xl_task = args.task
    single_label = False
    if args.task == 'ST-MEM':
        ptb_xl_task = 'superdiagnostic'
        single_label = True

    # Load labels
    logger.debug(f'setting up labels for task `{args.task}`')
    labels_df = PTB_XL.load_raw_labels(args.data_dir)
    labels_df = PTB_XL.compute_label_aggregations(labels_df, args.data_dir, ptb_xl_task)

    # Load data
    logger.debug(f'loading data from {dump_file}')
    channel_size = PTB_XL.record_duration * encoder_config.sampling_frequency

    x = load_data_dump(
        dump_file=dump_file,
        transform=PreprocessECG(
            channel_size=channel_size,
            remove_baseline_wander=False),
        processes=num_cpus)

    x, labels_df, y, class_names = PTB_XL.select_data(x, labels_df, ptb_xl_task, min_samples=0)
    if single_label:
        single_label_mask = y.sum(axis=1) == 1
        x, labels_df, y = x[single_label_mask], labels_df[single_label_mask], y[single_label_mask]
    y = torch.from_numpy(y).float()
    num_classes = y.shape[1]

    # PTB-XL stratified splits
    val_mask = (labels_df.strat_fold == args.val_fold).to_numpy()
    test_mask = (labels_df.strat_fold == args.test_fold).to_numpy()
    train_mask = ~(val_mask | test_mask)

    logger.debug(f'splits: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}')

    # Normalize data
    mean = np.mean(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
    std = np.std(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
    transforms.normalize_(x, mean_std=(mean, std))
    x.clip(-5, 5, out=x)

    # Ensure matching channels
    channel_order = get_channel_order(PTB_XL.channels, encoder_config.channels)
    x = x[:, :, channel_order]

    logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

    # Initialize eval config
    eval_config = configs.eval.Config(**eval_config_dict, num_classes=num_classes, depth=encoder_config.depth)

    # Cropping setup
    if eval_config.crop_duration is not None:
        crop_size = int(eval_config.crop_duration * encoder_config.sampling_frequency)
        crop_stride = int(eval_config.crop_stride * encoder_config.sampling_frequency) if eval_config.crop_stride else crop_size
    else:
        crop_size = None
        crop_stride = None

    # Data loaders
    train_loader = DataLoader(
        dataset=TensorDataset(
            data=x[train_mask],
            labels=y[train_mask],
            transform=TrainTransformECG(crop_size=crop_size)),
        batch_size=eval_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    def cycle(dataloader):
        while True:
            yield from dataloader

    train_iterator = cycle(train_loader)

    val_loader = DataLoader(
        dataset=TensorDataset(
            data=x[val_mask],
            labels=y[val_mask],
            transform=EvalTransformECG(crop_size=crop_size, crop_stride=crop_stride)),
        batch_size=eval_config.batch_size,
        num_workers=2)

    test_loader = DataLoader(
        dataset=TensorDataset(
            data=x[test_mask],
            labels=y[test_mask],
            transform=EvalTransformECG(crop_size=crop_size, crop_stride=crop_stride)),
        batch_size=eval_config.batch_size,
        num_workers=2)

    # LR schedule
    lr_schedule = cosine_schedule(
        total_steps=eval_config.steps,
        start_value=eval_config.learning_rate,
        final_value=eval_config.final_learning_rate,
        warmup_steps=eval_config.learning_rate_warmup_steps,
        warmup_start_value=1e-6)

    # Build model
    encoder = VisionTransformer(
        dim=encoder_config.dim,
        depth=encoder_config.depth,
        num_heads=encoder_config.num_heads,
        num_channels=encoder_config.num_channels,
        channel_size=encoder_config.channel_size,
        patch_size=encoder_config.patch_size,
        mlp_ratio=encoder_config.mlp_ratio,
        qkv_bias=encoder_config.qkv_bias,
        bias=encoder_config.bias,
        dropout=encoder_config.dropout,
        attn_dropout=encoder_config.attn_dropout,
        norm_eps=encoder_config.norm_eps,
        layer_scale_eps=encoder_config.layer_scale_eps,
        num_registers=encoder_config.num_registers,
        keep_registers=eval_config.use_register,
        use_sdp_kernel=using_cuda)

    model = ViTClassifier(encoder, eval_config, use_sdp_kernel=using_cuda).to(device)
    optimizer = model.get_optimizer(fused=using_cuda)

    # Load pretrained weights
    if model_state_dict:
        incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
        for key in incompatible_keys.missing_keys:
            logger.debug(f'missing {key} in the encoder checkpoint')
        for key in incompatible_keys.unexpected_keys:
            logger.debug(f'unexpected {key} in the encoder checkpoint')

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = args.run_name or f'{args.task}_{path.basename(args.encoder).replace(".pt", "")}'
        wandb.init(
            entity=args.entity,
            project='MyJEPA-ECG-Finetune',
            name=run_name,
            config={
                'task': args.task,
                'encoder': args.encoder,
                'val_fold': args.val_fold,
                'test_fold': args.test_fold,
                'num_classes': num_classes,
                'class_names': class_names,
                'train_samples': int(train_mask.sum()),
                'val_samples': int(val_mask.sum()),
                'test_samples': int(test_mask.sum()),
                'seed': args.seed,
                **asdict(eval_config),
                **asdict(encoder_config),
            }
        )
        wandb.watch(model, log='gradients', log_freq=100)
        
        # Log pretrained embeddings visualization
        logger.info('creating pretrained embedding visualization...')
        try:
            pretrained_fig = create_embedding_plot(
                model=model,
                dataloader=val_loader,
                device=device,
                class_names=class_names,
                title=f'{args.task.upper()} - Pretrained Embeddings (t-SNE)',
                num_samples=min(1000, val_mask.sum()),
                single_label=single_label
            )
            wandb.log({'embeddings/pretrained': wandb.Image(pretrained_fig)}, step=0)
            plt.close(pretrained_fig)
            logger.info('pretrained embedding plot logged to wandb')
        except Exception as e:
            logger.warning(f'failed to create pretrained embedding plot: {e}')

    # Training
    step_time = AverageMeter()
    train_loss = AverageMeter()
    best_val_auc = float('-inf')
    best_val_predictions, val_targets = None, None
    best_step, best_chkpt = None, None

    logger.info(f'starting finetuning on {args.task} task with {num_classes} classes')

    for step in range(eval_config.steps):
        step_start = time()
        lr = next(lr_schedule)
        update_learning_rate_(optimizer, lr)

        # Forward pass
        x_batch, y_batch = (tensor.to(device) for tensor in next(train_iterator))
        with auto_mixed_precision:
            logits = model(x_batch)
            if single_label:
                loss = F.cross_entropy(logits, y_batch)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)

        # Backward pass
        loss.backward()
        if eval_config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), eval_config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track metrics
        step_end = time()
        step_time.update(step_end - step_start)
        train_loss.update(loss.item())

        # Wandb logging
        if use_wandb and (step + 1) % 10 == 0:
            wandb.log({
                'train/loss': train_loss.value,
                'train/lr': lr,
                'train/step_time': step_time.value,
                **get_gpu_stats(),
            }, step=step + 1)

        # Evaluation
        if (step + 1) % eval_config.checkpoint_interval == 0:
            val_logits, val_targets_list = [], []
            model.eval()
            with torch.inference_mode():
                for batch in val_loader:
                    x_val, y_val = (tensor.to(device) for tensor in batch)
                    if eval_config.crop_duration is not None:
                        batch_size, num_crops, num_channels, channel_size = x_val.size()
                        x_val = x_val.reshape(-1, num_channels, channel_size)
                    logits = model(x_val)
                    if eval_config.crop_duration is not None:
                        logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
                        logits = logits.mean(dim=1)
                    val_logits.append(logits.clone())
                    val_targets_list.append(y_val.clone())
            model.train()

            if single_label:
                val_predictions = torch.cat(val_logits).softmax(dim=1).cpu().numpy()
            else:
                val_predictions = torch.cat(val_logits).sigmoid().cpu().numpy()
            val_targets = torch.cat(val_targets_list).cpu().numpy()
            
            val_auc = roc_auc_score(y_true=val_targets, y_score=val_predictions, average='macro')
            new_best_val_auc = val_auc > best_val_auc

            if new_best_val_auc:
                best_val_auc = val_auc
                best_val_predictions = val_predictions
                best_step = step
                best_chkpt = copy.deepcopy(model.state_dict())

            logger.info(f'[{step + 1:06d}] '
                        f'{"(*)" if new_best_val_auc else "   "} '
                        f'step_time {step_time.value:.4f} '
                        f'train_loss {train_loss.value:.4f} '
                        f'val_auc {val_auc:.4f}')

            if use_wandb:
                log_dict = {
                    'val/auc': val_auc,
                    'val/best_auc': best_val_auc,
                    'train/avg_loss': train_loss.value,
                    'train/avg_step_time': step_time.value,
                }
                try:
                    for i, name in enumerate(class_names):
                        class_auc = roc_auc_score(val_targets[:, i], val_predictions[:, i])
                        log_dict[f'val/auc_{name}'] = class_auc
                except:
                    pass
                wandb.log(log_dict, step=step + 1)

            step_time = AverageMeter()
            train_loss = AverageMeter()

            # Save checkpoint
            torch.save({
                'model': best_chkpt,
                'encoder_config': asdict(encoder_config),
                'eval_config': asdict(eval_config),
                'preprocess': {'mean': torch.from_numpy(mean.squeeze()),
                               'std': torch.from_numpy(std.squeeze())},
                'task': ptb_xl_task,
                'best_step': best_step,
                'best_val_auc': best_val_auc,
            }, path.join(args.out, f'{args.task}_best_chkpt.pt'))

            # Early stopping
            if step - best_step >= eval_config.early_stopping_patience:
                logger.info('stopping training early because validation AUC does not improve')
                if use_wandb:
                    wandb.log({'early_stopping': True, 'early_stopping_step': step + 1}, step=step + 1)
                break

    # Test evaluation
    logger.info('loading best model checkpoint')
    model.load_state_dict(best_chkpt)

    test_logits, test_targets_list = [], []
    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            x_test, y_test = (tensor.to(device) for tensor in batch)
            if eval_config.crop_duration is not None:
                batch_size, num_crops, num_channels, channel_size = x_test.size()
                x_test = x_test.reshape(-1, num_channels, channel_size)
            logits = model(x_test)
            if eval_config.crop_duration is not None:
                logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
                logits = logits.mean(dim=1)
            test_logits.append(logits.clone())
            test_targets_list.append(y_test.clone())

    if single_label:
        test_predictions = torch.cat(test_logits).softmax(dim=1).cpu().numpy()
    else:
        test_predictions = torch.cat(test_logits).sigmoid().cpu().numpy()
    test_targets = torch.cat(test_targets_list).cpu().numpy()
    
    test_auc = roc_auc_score(y_true=test_targets, y_score=test_predictions, average='macro')
    logger.info(f'test_auc {test_auc:.4f}')

    if use_wandb:
        log_dict = {
            'test/auc': test_auc,
            'test/best_step': best_step + 1,
        }
        try:
            for i, name in enumerate(class_names):
                class_auc = roc_auc_score(test_targets[:, i], test_predictions[:, i])
                log_dict[f'test/auc_{name}'] = class_auc
        except:
            pass
        wandb.log(log_dict)
        
        # Log finetuned embeddings visualization
        logger.info('creating finetuned embedding visualization...')
        try:
            finetuned_fig = create_embedding_plot(
                model=model,
                dataloader=val_loader,
                device=device,
                class_names=class_names,
                title=f'{args.task.upper()} - Finetuned Embeddings (t-SNE)',
                num_samples=min(1000, val_mask.sum()),
                single_label=single_label
            )
            wandb.log({'embeddings/finetuned': wandb.Image(finetuned_fig)})
            plt.close(finetuned_fig)
            logger.info('finetuned embedding plot logged to wandb')
        except Exception as e:
            logger.warning(f'failed to create finetuned embedding plot: {e}')
        
        wandb.finish()

    # Save predictions
    np.savez(path.join(args.out, f'{args.task}_predictions.npz'),
             val_targets=val_targets, val_predictions=best_val_predictions,
             test_targets=test_targets, test_predictions=test_predictions)


class PreprocessECG:
    def __init__(self, channel_size=None, remove_baseline_wander=False):
        self.channel_size = channel_size
        self.remove_baseline_wander = remove_baseline_wander

    def __call__(self, x):
        channel_size, num_channels = x.shape
        if self.remove_baseline_wander:
            x = transforms.highpass_filter(x, fs=PTB_XL.sampling_frequency)
        if self.channel_size is not None and self.channel_size != channel_size:
            x = transforms.resample(x, self.channel_size)
        return x


class TrainTransformECG:
    def __init__(self, crop_size=None):
        self.crop_size = crop_size

    def __call__(self, x):
        if self.crop_size is not None:
            x = transforms.random_crop(x, self.crop_size)
        x = x.transpose()
        x = torch.from_numpy(x).float()
        return x


class EvalTransformECG:
    def __init__(self, crop_size=None, crop_stride=None):
        self.crop_size = crop_size
        self.crop_stride = crop_stride or crop_size

    def __call__(self, x):
        if self.crop_size is not None:
            x = strided_crops(x, self.crop_size, self.crop_stride)
            x = np.swapaxes(x, 1, 2)
        else:
            x = x.transpose()
        x = torch.from_numpy(x).float()
        return x


def strided_crops(x, size, stride):
    channel_size, num_channels = x.shape
    crop_starts = range(0, channel_size - size + 1, stride)
    num_crops = len(crop_starts)
    x_ = np.empty((num_crops, size, num_channels), dtype=x.dtype)
    for i, start in enumerate(crop_starts):
        x_[i] = x[start:start + size]
    return x_


if __name__ == '__main__':
    main()

