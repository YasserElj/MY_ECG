# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from models.vit_nepa_ecg import ViTNepaECGForPreTraining, ViTNepaECGConfig


logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """Dataset for ECG signals from .npy file."""
    
    def __init__(
        self,
        npy_path: str,
        mean: np.ndarray,
        std: np.ndarray,
        crop_size: int = 5000,
        is_training: bool = True,
        max_samples: Optional[int] = None,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Args:
            npy_path: Path to .npy file with shape (N, signal_length, num_channels)
            mean: Mean for normalization, shape (1, 1, num_channels) or (num_channels,)
            std: Std for normalization, shape (1, 1, num_channels) or (num_channels,)
            crop_size: Size of temporal crop (default: 5000 samples = 10s at 500Hz)
            is_training: If True, use random crops; if False, use fixed crops
            max_samples: Maximum number of samples to use (for debugging)
            indices: Specific indices to use from the dataset
        """
        self.data = np.load(npy_path, mmap_mode='r')
        self.mean = mean.squeeze() if mean.ndim > 1 else mean
        self.std = std.squeeze() if std.ndim > 1 else std
        self.crop_size = crop_size
        self.is_training = is_training
        
        # Ensure mean/std have correct shape
        if self.mean.ndim == 0:
            self.mean = np.array([self.mean] * self.data.shape[-1])
        if self.std.ndim == 0:
            self.std = np.array([self.std] * self.data.shape[-1])
        
        # Use specific indices if provided
        if indices is not None:
            self.indices = indices
        else:
            # Limit samples if specified
            if max_samples is not None:
                self.indices = np.arange(min(max_samples, len(self.data)))
            else:
                self.indices = np.arange(len(self.data))
        
        logger.info(f"Loaded ECG dataset: {len(self.indices)} samples, shape {self.data.shape}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        
        # Load signal: shape (signal_length, num_channels)
        x = self.data[data_idx].copy().astype(np.float32)
        
        # Transpose to (num_channels, signal_length) for Conv1d
        if x.shape[0] > x.shape[1]:
            x = x.T
        
        # Normalize
        x = (x - self.mean[:, None]) / (self.std[:, None] + 1e-6)
        
        # Clip to [-5, 5]
        x = np.clip(x, -5, 5)
        
        # Temporal crop
        signal_length = x.shape[-1]
        if signal_length > self.crop_size:
            if self.is_training:
                # Random crop for training
                start = random.randint(0, signal_length - self.crop_size)
            else:
                # Center crop for validation
                start = (signal_length - self.crop_size) // 2
            x = x[..., start:start + self.crop_size]
        elif signal_length < self.crop_size:
            # Pad if too short (shouldn't happen with 5000 samples)
            pad_size = self.crop_size - signal_length
            x = np.pad(x, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        
        # Convert to tensor: (num_channels, signal_length)
        return {"pixel_values": torch.from_numpy(x)}


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to ECG data loading."""
    
    dataset_path: str = field(
        metadata={"help": "Path to .npy file containing ECG signals (shape: N, signal_length, num_channels)"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging: truncate number of training examples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging: truncate number of eval examples"}
    )
    train_val_split: float = field(
        default=0.05,
        metadata={"help": "Fraction of data to use for validation"}
    )
    crop_size: int = field(
        default=5000,
        metadata={"help": "Temporal crop size in samples (5000 = 10s at 500Hz)"}
    )


@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    
    config_name: str = field(
        metadata={"help": "Path to config.json file (e.g., configs/pretrain/nepa-small-ecg)"}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model (for resuming training)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code (required for custom models)"}
    )
    embed_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for embedding layer parameters"}
    )


class EnhancedTrainer(Trainer):
    """Trainer with embedding-specific learning rate support."""
    
    def __init__(
        self,
        *args,
        embed_lr=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embed_lr = embed_lr
    
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        
        if self.embed_lr is None:
            return super().create_optimizer()
        
        # Separate learning rate for embeddings
        from transformers.trainer_pt_utils import get_parameter_names
        
        decay_params = get_parameter_names(self.model, [torch.nn.LayerNorm])
        embed_params = set(
            f"vit_nepa.embeddings.{n}"
            for n, _ in self.model.vit_nepa.embeddings.named_parameters()
        )
        
        wd = self.args.weight_decay
        base_lr = self.args.learning_rate
        
        groups = [
            {"params": [], "weight_decay": wd, "lr": self.embed_lr},
            {"params": [], "weight_decay": 0.0, "lr": self.embed_lr},
            {"params": [], "weight_decay": wd, "lr": base_lr},
            {"params": [], "weight_decay": 0.0, "lr": base_lr},
        ]
        
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_decay = name in decay_params
            is_embed = name in embed_params
            
            if is_embed and is_decay:
                groups[0]["params"].append(p)
            elif is_embed and not is_decay:
                groups[1]["params"].append(p)
            elif (not is_embed) and is_decay:
                groups[2]["params"].append(p)
            else:
                groups[3]["params"].append(p)
        
        optimizer_grouped_parameters = [g for g in groups if g["params"]]
        
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def compute_normalization_stats(npy_path: str, train_indices: np.ndarray):
    """Compute mean and std from training data only."""
    logger.info("Computing normalization statistics from training data...")
    data = np.load(npy_path, mmap_mode='r')
    
    # Sample a subset for efficiency (use first 10k samples)
    sample_size = min(10000, len(train_indices))
    sample_indices = train_indices[:sample_size]
    
    sample_data = data[sample_indices].astype(np.float32)
    
    # Check for NaN/Inf values and replace them
    if np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data)):
        logger.warning("Found NaN/Inf values in data, replacing with 0")
        sample_data = np.nan_to_num(sample_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Data shape: (N, signal_length, num_channels)
    # Compute mean and std per channel across all samples and time steps
    # Result should be (1, 1, num_channels)
    mean = np.mean(sample_data, axis=(0, 1), keepdims=True, dtype=np.float32)
    std = np.std(sample_data, axis=(0, 1), keepdims=True, dtype=np.float32)
    
    # Avoid division by zero - ensure std is at least 1e-6
    std = np.maximum(std, 1e-6)
    
    # Check if we still have NaN (shouldn't happen after nan_to_num, but just in case)
    if np.any(np.isnan(mean)) or np.any(np.isnan(std)):
        logger.error("NaN values in normalization stats! Using default values.")
        num_channels = sample_data.shape[-1]
        mean = np.zeros((1, 1, num_channels), dtype=np.float32)
        std = np.ones((1, 1, num_channels), dtype=np.float32)
    
    logger.info(f"Normalization stats - Mean: {mean.squeeze()}, Std: {std.squeeze()}")
    return mean, std


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load data to determine split
    data = np.load(data_args.dataset_path, mmap_mode='r')
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.seed(training_args.seed)
    np.random.shuffle(indices)
    
    val_size = int(num_samples * data_args.train_val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    logger.info(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Compute normalization from training data only
    mean, std = compute_normalization_stats(data_args.dataset_path, train_indices)
    
    # Create datasets
    train_dataset = ECGDataset(
        data_args.dataset_path,
        mean,
        std,
        crop_size=data_args.crop_size,
        is_training=True,
        max_samples=data_args.max_train_samples,
        indices=train_indices,
    )
    
    val_dataset = ECGDataset(
        data_args.dataset_path,
        mean,
        std,
        crop_size=data_args.crop_size,
        is_training=False,
        max_samples=data_args.max_eval_samples,
        indices=val_indices,
    )
    
    # Load config and model
    config = ViTNepaECGConfig.from_pretrained(
        model_args.config_name,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    if model_args.model_name_or_path:
        model = ViTNepaECGForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = ViTNepaECGForPreTraining(config)
    
    # Collate function
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        return {"pixel_values": pixel_values}
    
    # Initialize trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        embed_lr=model_args.embed_lr,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
