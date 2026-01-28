import argparse
import dataclasses
import logging.config
import pprint
import os
import random
from contextlib import nullcontext
from os import path, makedirs
from time import time
import wandb

import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import configs
from data import transforms, utils as datautils
from data.datasets import (
  DATASETS,
  CODE15,
  StPetersburg,
  PTB_XL
)
from data.masks import MaskCollator
from data.utils import (
  TensorDataset,
  VariableTensorDataset,
  DatasetRouter
)
from models import JEPA
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
parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
parser.add_argument('--wandb-project', default='Physio-JEPA-ECG', help='wandb project name')
parser.add_argument('--wandb-entity', default=None, help='wandb entity (team) name')
parser.add_argument('--run-name', default=None, help='wandb run name')
parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
args = parser.parse_args()

# --- Dataset Statistics Updates ---
CODE15.mean = [0.000] * len(CODE15.channels)
CODE15.std = [0.488, 0.450, 0.437, 0.416, 0.405, 0.370,
              0.548, 0.639, 0.719, 0.695, 0.676, 0.639]
StPetersburg.mean = [0.000] * len(StPetersburg.channels)
StPetersburg.std = [0.132, 0.370, 0.353, 0.215, 0.191, 0.356,
                    0.234, 0.320, 0.328, 0.290, 0.317, 0.337]
PTB_XL.mean = [-0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
               0.000, -0.001, -0.002, -0.001, -0.001, -0.001]
PTB_XL.std = [0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
              0.235, 0.338, 0.335, 0.299, 0.294, 0.242]


def setup_distributed():
    """Initializes the distributed process group."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        print(f"Initialized Distributed Process: Rank {rank}/{world_size}, Local Rank {local_rank}")
        return device, rank, world_size, True
    else:
        print("Distributed environment variables not found. Running in single process mode.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1, False


def seed_worker(worker_id):
    """
    Seeds dataloader workers based on the torch initial seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def visualize_embeddings(model, batch_x, step, device):
    """Runs PCA and t-SNE on the current batch and logs to WandB."""
    model.eval()
    with torch.no_grad():
        # Handle DDP wrapper or compiled model
        if hasattr(model, 'module'):
            inner_model = model.module
        else:
            inner_model = model
            
        if hasattr(inner_model, 'encoder'):
            encoder = inner_model.encoder
        elif hasattr(inner_model, '_orig_mod'):
            encoder = inner_model._orig_mod.encoder
        else:
            return 

        embeddings = encoder(batch_x)
        embeddings = embeddings.mean(dim=1).cpu().numpy()

    model.train()

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # t-SNE (Fixed argument name: n_iter -> max_iter)
    n_samples = embeddings.shape[0]
    perp = min(30, n_samples - 1)
    if perp < 5: 
        return 
        
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, init='pca', learning_rate='auto')
    tsne_result = tsne.fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=15, c='tab:blue')
    ax1.set_title(f"PCA (Step {step})")
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, s=15, c='tab:orange')
    ax2.set_title(f"t-SNE (Step {step})")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Embedding Distribution @ Step {step}\n(Local Batch Size: {n_samples})")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    
    wandb.log({"embeddings/distribution": wandb.Image(image)}, step=step)
    
    plt.close(fig)
    buf.close()


def main():
  # 1. Setup Distributed Environment
  device, rank, world_size, is_distributed = setup_distributed()
  
  if rank == 0:
      makedirs(args.out, exist_ok=True)
      logging.config.fileConfig('logging.ini')
      
  logger = logging.getLogger('app')
  # Suppress logging on non-master ranks
  if rank != 0:
      logger.setLevel(logging.WARNING)

  using_cuda = device.type == 'cuda'
  num_cpus = get_cpu_count()
  
  if rank == 0:
      logger.debug(f'using {device} accelerator and {num_cpus} CPUs (World Size: {world_size})')

  if using_cuda:
    if rank == 0: logger.debug('TF32 tensor cores are enabled')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # For strict reproducibility, set benchmark to False, but it slows down training.
    # We keep it True for performance, as seeding handles logic reproducibility.
    torch.backends.cudnn.benchmark = True

  # Mixed Precision Setup (Updated to fix deprecation warning)
  if args.amp == 'float32' or not using_cuda:
    if rank == 0: logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif args.amp == 'bfloat16':
    if rank == 0: logger.debug('using bfloat16 with AMP')
    # Updated API
    auto_mixed_precision = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')

  # 2. Config Loading
  # Resolve config file path
  if not path.isfile(args.config):
    config_file = path.join(path.dirname(configs.pretrain.__file__),  f'{args.config}.yaml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {args.config}')
    args.config = config_file
  
  # Load config file
  config_dict = configs.load_config_file(args.config)
  
  if args.chkpt:
    if rank == 0: logger.debug(f'resuming from checkpoint {args.chkpt}')
    chkpt = torch.load(args.chkpt, map_location='cpu') 
    # Start with checkpoint config
    config = configs.pretrain.Config(**chkpt['config'])
    # Override batch_size and gradient_accumulation_steps from config file if present
    if 'batch_size' in config_dict:
      config.batch_size = config_dict['batch_size']
      if rank == 0: logger.debug(f'overriding batch_size from config file: {config.batch_size}')
    if 'gradient_accumulation_steps' in config_dict:
      config.gradient_accumulation_steps = config_dict['gradient_accumulation_steps']
      if rank == 0: logger.debug(f'overriding gradient_accumulation_steps from config file: {config.gradient_accumulation_steps}')
  else:
    config = configs.pretrain.Config(**config_dict)
    if rank == 0:
        logger.debug(f'loading configuration file from {args.config}:\n'
                     f'{pprint.pformat(config_dict, compact=True, sort_dicts=False, width=120)}')
    chkpt = None

  # 3. WandB Init (Only on Rank 0)
  if args.wandb and rank == 0:
    wandb.init(
      project=args.wandb_project,
      entity=args.wandb_entity,
      name=args.run_name,
      config=dataclasses.asdict(config),
      resume="allow" if args.chkpt else None
    )

  # 4. Data Loading
  dump_files = {}
  for data_arg in args.data:
    dataset_name, *maybe_dump_file = data_arg.split('=', 1)
    if not maybe_dump_file:
      raise ValueError('Dataset pair must have following format: dataset=path/to/data.npy')
    dump_file, = maybe_dump_file
    dump_files[dataset_name] = dump_file

  datasets = {}
  for dataset_name, weight in config.datasets.items():
    if dataset_name not in dump_files:
       continue 
    dump_file = dump_files[dataset_name]
    
    if rank == 0: logger.debug(f'loading {dataset_name} from {dump_file}')
    
    dataset_cls = DATASETS[dataset_name]
    resample_ratio = config.sampling_frequency / dataset_cls.sampling_frequency
    channel_order = datautils.get_channel_order(dataset_cls.channels, config.channels)
    mean = np.array([dataset_cls.mean], dtype=np.float16)
    std = np.array([dataset_cls.std], dtype=np.float16)
    _, ext = path.splitext(dump_file)

    preprocess = PreprocessECG(
            mean_std=(mean, std),
            resample_ratio=resample_ratio,
            channel_order=channel_order)
    transform = TransformECG(crop_size=config.channel_size)

    if ext == '.npy':
      data = datautils.load_data_dump(dump_file=dump_file, transform=preprocess, processes=num_cpus)
      dataset = TensorDataset(data=data, transform=transform)
    elif ext == '.npz':
      data_tuple = load_variable_data_dump(dump_file=dump_file, min_channel_size=config.channel_size,
                                           transform=preprocess, processes=num_cpus)
      dataset = VariableTensorDataset(*data_tuple, transform=transform)
    else:
      raise ValueError(f'Unsupported dataset format: {dump_file}')
    
    datasets[dataset_name] = (dataset, weight)

  if rank == 0:
      logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  # 5. Distributed Batch Size & Seeding
  if config.batch_size % world_size != 0:
      raise ValueError(f"Batch size ({config.batch_size}) must be divisible by world size ({world_size})")
  
  batch_size_per_gpu = config.batch_size // world_size

  # SEEDING: Ensure each rank has a unique but deterministic seed
  # We add rank to the seed so GPUs process different data, but consistently.
  global_seed = args.seed + rank
  torch.manual_seed(global_seed)
  np.random.seed(global_seed)
  random.seed(global_seed)

  train_loader = DataLoader(
    dataset=DatasetRouter(datasets.values()),
    batch_size=batch_size_per_gpu, 
    pin_memory=using_cuda,
    collate_fn=MaskCollator(
      patch_size=config.patch_size,
      min_block_size=config.min_block_size,
      min_keep_ratio=config.min_keep_ratio,
      max_keep_ratio=config.max_keep_ratio),
    num_workers=2,
    worker_init_fn=seed_worker # This will use the torch seed set above
    )

  def map_to_device(data_iterator, device=None):
    for batch in data_iterator:
      yield tuple(x.to(device, non_blocking=using_cuda) for x in batch)

  def prefetch_batch(data_iterator):
    prefetched_batch = next(data_iterator)
    for next_batch in data_iterator:
      yield prefetched_batch
      prefetched_batch = next_batch
    yield prefetched_batch

  train_iterator = iter(train_loader)
  train_iterator = map_to_device(train_iterator, device=device)
  train_iterator = prefetch_batch(train_iterator)

  # 6. Model Setup & DDP Wrapping
  if chkpt is not None:
    start_step = chkpt['step']
  else:
    start_step = 0

  momentum_schedule = linear_schedule(
    total_steps=config.steps,
    start_value=config.encoder_momentum,
    final_value=config.final_encoder_momentum,
    step=start_step)
  lr_schedule = cosine_schedule(
    total_steps=config.steps,
    start_value=config.learning_rate,
    final_value=config.final_learning_rate,
    warmup_steps=config.learning_rate_warmup_steps,
    warmup_start_value=1e-6,
    step=start_step)
  wd_schedule = cosine_schedule(
    total_steps=config.steps,
    start_value=config.weight_decay,
    final_value=config.final_weight_decay,
    step=start_step)

  model = JEPA(
    config=config,
    momentum_schedule=momentum_schedule,
    use_sdp_kernel=using_cuda
  ).to(device)

  original_model = model 

  if chkpt is not None:
    model.load_state_dict(chkpt['model'])

  if args.compile:
    model = torch.compile(model)

  if is_distributed:
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    original_model = model.module

  optimizer = original_model.get_optimizer(fused=using_cuda)
  if chkpt is not None:
    optimizer.load_state_dict(chkpt['optimizer'])

  step_time = AverageMeter()
  train_loss = AverageMeter()

  for step in range(start_step, config.steps):
    step_start = time()
    
    current_lr = next(lr_schedule)
    current_wd = next(wd_schedule)
    update_learning_rate_(optimizer, current_lr)
    update_weight_decay_(optimizer, current_wd)
    
    batch_loss = 0.
    for _ in range(config.gradient_accumulation_steps):
      x, mask_encoder, mask_predictor = next(train_iterator)
      with auto_mixed_precision:
        loss = model(x, mask_encoder, mask_predictor)
        loss = loss / config.gradient_accumulation_steps
      loss.backward()
      batch_loss += loss.item()
    
    if config.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    
    optimizer.step()
    train_loss.update(batch_loss)
    
    optimizer.zero_grad(set_to_none=True)
    step_end = time()
    step_time.update(step_end - step_start)

    # 7. Visualization & Logging (Rank 0 only)
    if rank == 0:
        # NOTE: Visualizations occur here.
        if (step + 1) % 1000 == 0 and args.wandb:
            visualize_embeddings(model, x, step + 1, device)
            
        if (step + 1) % 100 == 0:
            logger.info(f'[{step + 1:06d}] '
                        f'step_time {step_time.value:.4f} '
                        f'train_loss {train_loss.value:.4f}')
        
            if args.wandb:
                wandb.log({
                    'train/loss': train_loss.value,
                    'train/step_time': step_time.value,
                    'train/lr': current_lr,
                    'train/weight_decay': current_wd,
                }, step=step + 1)
            
            step_time = AverageMeter()
            train_loss = AverageMeter()
            
        if (step + 1) % config.checkpoint_interval == 0:
            torch.save({
                'model': original_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': dataclasses.asdict(config),
                'step': step + 1,
            }, path.join(args.out, f'chkpt_{step + 1}.pt'))

def load_variable_data_dump(dump_file, min_channel_size, transform=None, processes=None):
  data = datautils.load_variable_data_dump(dump_file, transform=transform, processes=processes)
  data = [x for x in data if len(x) >= min_channel_size]
  sizes = np.array([len(x) for x in data])
  # starts is not used in return, removing to avoid unused variable warning if strict
  data = np.concatenate(data)
  return data, sizes

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
