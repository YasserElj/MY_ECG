import argparse
import copy
import dataclasses
import logging.config
import pprint
from contextlib import nullcontext
from os import path, makedirs
from time import time
import io

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

import wandb
import configs
from data import transforms, utils as datautils
from data.datasets import PTB_XL
from data.utils import TensorDataset
from models import VisionTransformer, FactorizedViT, GroupedViT, ViTClassifier
from utils.monitoring import AverageMeter, get_memory_usage, get_cpu_count
from utils.schedules import update_learning_rate_, cosine_schedule

TASKS = (
  'all',
  'diagnostic',
  'subdiagnostic',
  'superdiagnostic',
  'form',
  'rhythm',
  # custom tasks
  'ST-MEM',  # Na et al. (2024)
)
FOLDS = tuple(range(1, 11))

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to data directory')
parser.add_argument('--encoder', required=True, help='path to checkpoint or config file')
parser.add_argument('--out', default='eval', help='output directory')
parser.add_argument('--config', default='linear', help='path to config file or config name')
parser.add_argument('--dump', help='path to dump file (.npy) with raw ECG signals')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'], help='automated mixed precision')
parser.add_argument('--task', choices=TASKS, default='all', help='task type')
parser.add_argument('--val-fold', choices=FOLDS, type=int, default=9, help='validation fold')
parser.add_argument('--test-fold', choices=FOLDS, type=int, default=10, help='test fold')
# WandB Arguments
parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
parser.add_argument('--wandb-project', default='Physio-JEPA-Finetune', help='wandb project name')
parser.add_argument('--wandb-entity', default=None, help='wandb entity (team) name')
parser.add_argument('--run-name', default=None, help='wandb run name')
args = parser.parse_args()


def visualize_embeddings(model, loader, step, device, title_suffix=""):
    """
    Extracts embeddings from the validation set (or a subset), 
    runs PCA/t-SNE, and logs colored plots to WandB.
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    
    # Collect a subset of data for visualization (e.g., up to 1024 samples)
    max_samples = 1024
    samples_count = 0
    
    with torch.no_grad():
        for batch in loader:
            x, y = (tensor.to(device) for tensor in batch)
            
            # Handle cropping if necessary (similar to validation loop)
            # We assume the loader provided passed EvalTransformECG
            if x.ndim == 4: # (Batch, Crops, Channels, Time)
                 batch_size, num_crops, num_channels, channel_size = x.size()
                 x = x.reshape(-1, num_channels, channel_size)
            
            # Extract features from the ENCODER directly
            # encoder returns (Batch, Sequence, Dim)
            features = model.encoder(x)
            
            # Global Average Pooling (Time dimension)
            # Shape: (Batch, Dim)
            features = features.mean(dim=1)
            
            # Aggregate crops if necessary
            if 'num_crops' in locals():
                features = features.reshape(batch_size, num_crops, -1).mean(dim=1)
            
            embeddings_list.append(features.cpu().numpy())
            labels_list.append(y.cpu().numpy())
            
            samples_count += x.shape[0] if 'num_crops' not in locals() else batch_size
            if samples_count >= max_samples:
                break
    
    embeddings = np.concatenate(embeddings_list, axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    
    model.train()

    # Determine Colors based on Task Labels
    # For multi-label, we pick the first active class (argmax) for visualization coloring
    # Shape of labels: (N, Num_Classes)
    color_indices = labels.argmax(axis=1)
    num_classes = labels.shape[1]

    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # 2. t-SNE
    n_samples = embeddings.shape[0]
    perp = min(30, n_samples - 1)
    if perp > 5:
        tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, init='pca', learning_rate='auto')
        tsne_result = tsne.fit_transform(embeddings)
    else:
        tsne_result = None

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create a colormap
    cmap = plt.get_cmap('tab10' if num_classes <= 10 else 'turbo', num_classes)
    
    # PCA Plot
    scatter1 = ax1.scatter(pca_result[:, 0], pca_result[:, 1], 
                           c=color_indices, cmap=cmap, alpha=0.7, s=20)
    ax1.set_title(f"PCA {title_suffix}")
    ax1.grid(True, alpha=0.3)
    if num_classes <= 20: # Only show legend if not too cluttered
        plt.colorbar(scatter1, ax=ax1, ticks=range(num_classes), label='Class Index')

    # t-SNE Plot
    if tsne_result is not None:
        scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                               c=color_indices, cmap=cmap, alpha=0.7, s=20)
        ax2.set_title(f"t-SNE {title_suffix}")
        ax2.grid(True, alpha=0.3)
        if num_classes <= 20:
             plt.colorbar(scatter2, ax=ax2, ticks=range(num_classes), label='Class Index')
    else:
        ax2.text(0.5, 0.5, "Not enough samples for t-SNE", ha='center')

    plt.suptitle(f"Embedding Clusters @ Step {step}\nColored by Task Label (Argmax)")
    plt.tight_layout()

    # 4. Log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    
    wandb.log({f"embeddings/{title_suffix}distribution": wandb.Image(image), "step": step})
    
    plt.close(fig)
    buf.close()


def main():
  makedirs(args.out, exist_ok=True)
  logging.config.fileConfig('logging.ini')
  logger = logging.getLogger('app')

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

  if args.amp == 'float32' or not using_cuda:  # don't use AMP on a CPU
    logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif args.amp == 'bfloat16':
    # bfloat16 preserves the range of float32, so it does not require scaling
    logger.debug('using bfloat16 with AMP')
    auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')

  if not path.isfile(args.config):
    # maybe config is the name of a default config file in configs/pretrain/
    config_file = path.join(path.dirname(configs.eval.__file__), f'{args.config}.yaml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {args.config}')
    args.config = config_file

  eval_config_dict = configs.load_config_file(args.config)
  logger.debug(f'loading configuration file from {args.config}\n'
               f'{pprint.pformat(eval_config_dict, compact=True, sort_dicts=False, width=120)}')

  # load checkpoint
  _, ext = path.splitext(args.encoder)
  if ext == '.yaml':
    logger.debug(f'loading encoder config from {args.encoder}')
    encoder_config_dict = configs.load_config_file(args.encoder)
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    model_state_dict = None
  else:
    logger.debug(f'loading encoder checkpoint from {args.encoder}')
    chkpt = torch.load(args.encoder, map_location='cpu')
    encoder_config_dict = chkpt['config']
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    if 'eval_config' in chkpt:  # continue fine-tuning the weights
      model_state_dict = chkpt['model']
    else:  # extract target encoder's weights from the checkpoint
      model_state_dict = {'encoder.' + k.removeprefix('target_encoder.'): v
                          for k, v in chkpt['model'].items()
                          if k.startswith('target_encoder.')}

  ptb_xl_task = args.task
  single_label = False
  if args.task == 'ST-MEM':
    ptb_xl_task = 'superdiagnostic'
    single_label = True

  # load labels
  logger.debug(f'setting up labels for task `{args.task}`')
  labels_df = PTB_XL.load_raw_labels(args.data_dir)
  labels_df = PTB_XL.compute_label_aggregations(labels_df, args.data_dir, ptb_xl_task)

  # load data
  logger.debug(f'loading data from {dump_file}')
  channel_size = PTB_XL.record_duration * encoder_config.sampling_frequency

  x = datautils.load_data_dump(
    dump_file=dump_file,
    transform=PreprocessECG(
      channel_size=channel_size,
      remove_baseline_wander=False),
    processes=num_cpus)

  x, labels_df, y, _ = PTB_XL.select_data(x, labels_df, ptb_xl_task, min_samples=0)
  if single_label:
    single_label_mask = y.sum(axis=1) == 1
    x, labels_df, y = x[single_label_mask], labels_df[single_label_mask], y[single_label_mask]
  y = torch.from_numpy(y).float()
  num_classes = y.shape[1]

  val_mask = (labels_df.strat_fold == args.val_fold).to_numpy()
  test_mask = (labels_df.strat_fold == args.test_fold).to_numpy()
  train_mask = ~(val_mask | test_mask)

  # normalize data
  mean = np.mean(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
  std = np.std(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
  transforms.normalize_(x, mean_std=(mean, std))
  x.clip(-5, 5, out=x)

  # ensure matching channels
  channel_order = datautils.get_channel_order(PTB_XL.channels, encoder_config.channels)
  x = x[:, :, channel_order]

  logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  # initialize configs
  eval_config = configs.eval.Config(**eval_config_dict, num_classes=num_classes)
  if eval_config.use_register and encoder_config.num_registers == 0:
    logger.debug('adding a randomly initialized register to the encoder')
    encoder_config = dataclasses.replace(encoder_config, num_registers=1)

  if eval_config.dropout != encoder_config.dropout:
    logger.debug('overriding encoder dropout')
    encoder_config = dataclasses.replace(encoder_config, dropout=eval_config.dropout)

  if encoder_config.layer_scale_eps == 0 and eval_config.layer_scale_eps > 0:
    logger.debug('adding LayerScale to the encoder')
    encoder_config = dataclasses.replace(encoder_config, layer_scale_eps=eval_config.layer_scale_eps)

  if eval_config.crop_duration is not None:
    crop_size = int(eval_config.crop_duration * encoder_config.sampling_frequency)
    if eval_config.crop_stride is not None:
      crop_stride = int(eval_config.crop_stride * encoder_config.sampling_frequency)
    else:
      crop_stride = crop_size
  else:
    crop_size = None
    crop_stride = None

  train_loader = DataLoader(
    dataset=TensorDataset(
      data=x[train_mask],
      labels=y[train_mask],
      transform=TrainTransformECG(
        crop_size=crop_size)),
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
      transform=EvalTransformECG(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=2)
  test_loader = DataLoader(
    dataset=TensorDataset(
      data=x[test_mask],
      labels=y[test_mask],
      transform=EvalTransformECG(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=2)

  # WANDB INIT
  if args.wandb:
      wandb.init(
          project=args.wandb_project,
          entity=args.wandb_entity,
          name=args.run_name,
          config={
              "encoder_config": dataclasses.asdict(encoder_config),
              "eval_config": dataclasses.asdict(eval_config),
              "task": args.task,
          }
      )

  # setup hyperparameter schedules
  lr_schedule = cosine_schedule(
    total_steps=eval_config.steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
    warmup_start_value=1e-6)

  # Detect architecture type from checkpoint config
  structure = encoder_config_dict.get('structure', 'standard')
  
  if structure == 'factorized':
    logger.debug('using FactorizedViT encoder (detected from checkpoint config)')
    # Factorized model doesn't have register tokens
    if not eval_config.attn_pooling and eval_config.use_register:
      logger.warning('FactorizedViT does not support register tokens. '
                     'Switching to mean pooling (use_register=False)')
      eval_config = dataclasses.replace(eval_config, use_register=False)
    encoder = FactorizedViT(
      config=encoder_config,
      keep_registers=False,
      use_sdp_kernel=using_cuda)
  elif structure == 'grouped':
    logger.debug('using GroupedViT encoder (detected from checkpoint config)')
    # Grouped model doesn't have register tokens
    if not eval_config.attn_pooling and eval_config.use_register:
      logger.warning('GroupedViT does not support register tokens. '
                     'Switching to mean pooling (use_register=False)')
      eval_config = dataclasses.replace(eval_config, use_register=False)
    encoder = GroupedViT(
      config=encoder_config,
      keep_registers=False,
      use_sdp_kernel=using_cuda)
  else:
    logger.debug('using standard VisionTransformer encoder')
    encoder = VisionTransformer(
      config=encoder_config,
      keep_registers=eval_config.use_register,
      use_sdp_kernel=using_cuda)
  
  model = ViTClassifier(encoder, eval_config, use_sdp_kernel=using_cuda).to(device)
  optimizer = model.get_optimizer(fused=using_cuda)

  if model_state_dict is not None:
    incompatible_keys = model.load_state_dict(model_state_dict, strict=False)
    for key in incompatible_keys.missing_keys:
      logger.debug(f'missing {key} in the encoder checkpoint')
    for key in incompatible_keys.unexpected_keys:
      logger.debug(f'unexpected {key} in the encoder checkpoint')

  step_time = AverageMeter()
  train_loss = AverageMeter()
  best_val_auc = float('-inf')
  best_val_predictions, val_targets = None, None
  best_step, best_chkpt = None, None

  for step in range(eval_config.steps):
    step_start = time()
    # update hyperparameters according to schedule
    update_learning_rate_(optimizer, next(lr_schedule))
    # forward pass
    x, y = (tensor.to(device) for tensor in next(train_iterator))
    with auto_mixed_precision:
      logits = model(x)
      if single_label:
        loss = F.cross_entropy(logits, y)
      else:
        loss = F.binary_cross_entropy_with_logits(logits, y)
    # backward pass
    loss.backward()
    if eval_config.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), eval_config.gradient_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # finalize train step
    step_end = time()
    step_time.update(step_end - step_start)
    train_loss.update(loss.item())
    
    # evaluation
    if (step + 1) % eval_config.checkpoint_interval == 0:
      # --- Visualization Start ---
      if args.wandb:
          # Visualize using Validation loader (or you can create a dedicated subset loader)
          visualize_embeddings(model, val_loader, step + 1, device, title_suffix=f"({args.task})")
      # --- Visualization End ---

      val_logits, val_targets = [], []
      model.eval()
      with torch.inference_mode():
        for batch in val_loader:
          x, y = (tensor.to(device) for tensor in batch)
          if eval_config.crop_duration is not None:
            batch_size, num_crops, num_channels, channel_size = x.size()
            x = x.reshape(-1, num_channels, channel_size)
          logits = model(x)
          if eval_config.crop_duration is not None:
            logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
            logits = logits.mean(dim=1)  # aggregate crop predictions
          val_logits.append(logits.clone())
          val_targets.append(y.clone())
      model.train()
      if single_label:
        val_predictions = torch.cat(val_logits).softmax(dim=1).cpu().numpy()
      else:
        val_predictions = torch.cat(val_logits).sigmoid().cpu().numpy()
      val_targets = torch.cat(val_targets).cpu().numpy()
      val_auc = roc_auc_score(
        y_true=val_targets,
        y_score=val_predictions,
        average='macro')
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

      if args.wandb:
          wandb.log({
              "train/loss": train_loss.value,
              "val/auc": val_auc,
              "val/best_auc": best_val_auc,
              "step": step + 1
          })

      step_time = AverageMeter()
      train_loss = AverageMeter()
      if step - best_step >= eval_config.early_stopping_patience:
        logging.info('stopping training early because validation AUC does not improve')
        break

  torch.save({
    'model': best_chkpt,
    'config': dataclasses.asdict(encoder_config),
    'eval_config': dataclasses.asdict(eval_config),
    'preprocess': {'mean': torch.from_numpy(mean.squeeze()),
                   'std': torch.from_numpy(std.squeeze())},
    'task': ptb_xl_task
  }, path.join(args.out, f'{args.task}_best_chkpt.pt'))

  # test model
  logger.info('loading best model checkpoint')
  model.load_state_dict(best_chkpt)

  test_logits, test_targets = [], []
  model.eval()
  with torch.inference_mode():
    for batch in test_loader:
      x, y = (tensor.to(device) for tensor in batch)
      if eval_config.crop_duration is not None:
        batch_size, num_crops, num_channels, channel_size = x.size()
        x = x.reshape(-1, num_channels, channel_size)
      logits = model(x)
      if eval_config.crop_duration is not None:
        logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
        logits = logits.mean(dim=1)  # aggregate crop predictions
      test_logits.append(logits.clone())
      test_targets.append(y.clone())
  if single_label:
    test_predictions = torch.cat(test_logits).softmax(dim=1).cpu().numpy()
  else:
    test_predictions = torch.cat(test_logits).sigmoid().cpu().numpy()
  test_targets = torch.cat(test_targets).cpu().numpy()
  test_auc = roc_auc_score(
      y_true=test_targets,
      y_score=test_predictions,
      average='macro')
  logger.info(f'test_auc {test_auc:.4f}')
  
  if args.wandb:
      wandb.log({"test/auc": test_auc})

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


class TrainTransformECG:   # called whenever dataloader accesses the data
  def __init__(self, crop_size=None):
    self.crop_size = crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


class EvalTransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size=None, crop_stride=None):
    self.crop_size = crop_size
    self.crop_stride = crop_stride or crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = strided_crops(x, self.crop_size, self.crop_stride)
      x = np.swapaxes(x, 1, 2)  # channels first
    else:
      x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


def strided_crops(x, size, stride):  # x: (channel_size, num_channels)
  channel_size, num_channels = x.shape
  crop_starts = range(0, channel_size - size + 1, stride)
  num_crops = len(crop_starts)
  x_ = np.empty((num_crops, size, num_channels), dtype=x.dtype)
  for i, start in enumerate(crop_starts):
    x_[i] = x[start:start + size]
  return x_


if __name__ == '__main__':
  main()