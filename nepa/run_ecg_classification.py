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

import ast
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from models.vit_nepa_ecg import ViTNepaECGForImageClassification, ViTNepaECGConfig


logger = logging.getLogger(__name__)


def load_ptbxl_labels(data_dir: str, task: str = "rhythm"):
    """Load PTB-XL labels for a specific task."""
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    
    # Parse SCP codes
    scp_codes = df["scp_codes"].apply(ast.literal_eval)
    
    # Define task-specific label mappings
    if task == "rhythm":
        # Rhythm labels
        rhythm_labels = [
            "NORM", "AFIB", "STACH", "SBRAD", "SARRH", "SVTACH", "PACE", "SVARR", "BIGU", "AFLT"
        ]
    elif task == "diagnostic":
        # Diagnostic labels (simplified - you may want to expand this)
        rhythm_labels = [
            "NORM", "MI", "STTC", "HYP", "CD", "IVCD", "LBBB", "RBBB", "LAFB", "IRBBB"
        ]
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create label matrix
    num_samples = len(df)
    num_labels = len(rhythm_labels)
    labels = np.zeros((num_samples, num_labels), dtype=np.float32)
    
    label_to_idx = {label: idx for idx, label in enumerate(rhythm_labels)}
    
    for idx, codes in enumerate(scp_codes):
        for code, value in codes.items():
            if code in label_to_idx and value > 0:
                labels[idx, label_to_idx[code]] = 1.0
    
    return labels, rhythm_labels, df["strat_fold"].values


class PTBXLECGDataset(Dataset):
    """Dataset for PTB-XL ECG signals."""
    
    def __init__(
        self,
        npy_path: str,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        crop_size: int = 5000,
        crop_stride: Optional[int] = None,
        is_training: bool = True,
    ):
        """
        Args:
            npy_path: Path to .npy file with shape (N, signal_length, num_channels)
            labels: Label matrix of shape (N, num_classes)
            indices: Which samples to use from the dataset
            mean: Mean for normalization
            std: Std for normalization
            crop_size: Size of temporal crop
            crop_stride: Stride for evaluation crops (if None, use crop_size)
            is_training: If True, use random crops; if False, use strided crops
        """
        self.data = np.load(npy_path, mmap_mode='r')
        self.labels = labels
        self.indices = indices
        self.mean = mean.squeeze() if mean.ndim > 1 else mean
        self.std = std.squeeze() if std.ndim > 1 else std
        self.crop_size = crop_size
        self.crop_stride = crop_stride or crop_size
        self.is_training = is_training
        
        # Ensure mean/std have correct shape
        if self.mean.ndim == 0:
            self.mean = np.array([self.mean] * self.data.shape[-1])
        if self.std.ndim == 0:
            self.std = np.array([self.std] * self.data.shape[-1])
    
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
        
        signal_length = x.shape[-1]
        
        if self.is_training:
            # Random crop for training
            if signal_length > self.crop_size:
                start = random.randint(0, signal_length - self.crop_size)
                x = x[..., start:start + self.crop_size]
            elif signal_length < self.crop_size:
                pad_size = self.crop_size - signal_length
                x = np.pad(x, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
            return {
                "pixel_values": torch.from_numpy(x),
                "labels": torch.from_numpy(self.labels[data_idx]).float(),
            }
        else:
            # Strided crops for evaluation
            crops = []
            for start in range(0, signal_length - self.crop_size + 1, self.crop_stride):
                crop = x[..., start:start + self.crop_size]
                crops.append(crop)
            
            if len(crops) == 0:
                # Fallback if signal is too short
                if signal_length < self.crop_size:
                    pad_size = self.crop_size - signal_length
                    x = np.pad(x, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
                    crops = [x]
                else:
                    # Center crop
                    start = (signal_length - self.crop_size) // 2
                    crops = [x[..., start:start + self.crop_size]]
            
            return {
                "pixel_values": torch.stack([torch.from_numpy(c) for c in crops]),
                "labels": torch.from_numpy(self.labels[data_idx]).float(),
            }


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to ECG data loading."""
    
    data_dir: str = field(
        metadata={"help": "Path to PTB-XL data directory (should contain ptbxl_database.csv and ptb-xl.npy)"}
    )
    task: str = field(
        default="rhythm",
        metadata={"help": "Task type: 'rhythm' or 'diagnostic'"}
    )
    val_fold: int = field(
        default=9,
        metadata={"help": "Validation fold (1-10)"}
    )
    test_fold: int = field(
        default=10,
        metadata={"help": "Test fold (1-10)"}
    )
    crop_duration: Optional[float] = field(
        default=2.5,
        metadata={"help": "Crop duration in seconds (at 500Hz, 2.5s = 1250 samples)"}
    )
    crop_stride: Optional[float] = field(
        default=1.25,
        metadata={"help": "Crop stride in seconds for evaluation"}
    )


@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model checkpoint"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to config.json file (if different from model)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code"}
    )
    is_causal: bool = field(
        default=False,
        metadata={"help": "Whether to use causal attention (False for bidirectional)"}
    )


class ECGTrainer(Trainer):
    """Trainer with support for strided crop averaging during evaluation."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Multi-label binary cross-entropy
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        # Handle strided crops: average predictions
        if "pixel_values" in inputs and inputs["pixel_values"].ndim == 4:
            # Shape: (batch_size, num_crops, num_channels, signal_length)
            batch_size, num_crops = inputs["pixel_values"].shape[:2]
            pixel_values = inputs["pixel_values"].reshape(-1, *inputs["pixel_values"].shape[2:])
            
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
            
            # Reshape and average
            logits = logits.reshape(batch_size, num_crops, -1)
            logits = logits.mean(dim=1)  # Average over crops
            
            loss = None
            if has_labels:
                labels = inputs["labels"]
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                loss = None
                if has_labels:
                    loss = self.compute_loss(model, inputs, return_outputs=False)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        if has_labels:
            labels = inputs["labels"]
        else:
            labels = None
        
        return (loss, logits, labels)


def compute_auc_metrics(eval_pred):
    """Compute AUC for multi-label classification."""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(torch.from_numpy(predictions)).numpy()
    
    # Compute macro-averaged AUC
    try:
        auc = roc_auc_score(labels, probs, average='macro')
    except ValueError:
        # Some labels might be all zeros or all ones
        auc = 0.5
    
    return {"auc": auc}


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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    )
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load PTB-XL labels
    labels, label_names, strat_folds = load_ptbxl_labels(data_args.data_dir, data_args.task)
    num_classes = len(label_names)
    logger.info(f"Loaded {len(labels)} samples with {num_classes} classes: {label_names}")
    
    # Create train/val/test splits
    train_mask = (strat_folds != data_args.val_fold) & (strat_folds != data_args.test_fold)
    val_mask = strat_folds == data_args.val_fold
    test_mask = strat_folds == data_args.test_fold
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # Load data and compute normalization from training set only
    npy_path = os.path.join(data_args.data_dir, "ptb-xl.npy")
    if not os.path.exists(npy_path):
        npy_path = os.path.join(data_args.data_dir, "..", "ptb-xl.npy")
    
    data = np.load(npy_path, mmap_mode='r')
    
    # Compute normalization stats from training data
    sample_size = min(10000, len(train_indices))
    sample_indices = train_indices[:sample_size]
    sample_data = data[sample_indices].astype(np.float32)
    mean = np.mean(sample_data, axis=(0, 1), keepdims=True, dtype=np.float32)
    std = np.std(sample_data, axis=(0, 1), keepdims=True, dtype=np.float32)
    
    logger.info(f"Normalization stats - Mean: {mean.squeeze()}, Std: {std.squeeze()}")
    
    # Convert crop duration to samples (assuming 500Hz)
    sampling_freq = 500
    crop_size = int(data_args.crop_duration * sampling_freq) if data_args.crop_duration else 5000
    crop_stride = int(data_args.crop_stride * sampling_freq) if data_args.crop_stride else crop_size
    
    # Create datasets
    train_dataset = PTBXLECGDataset(
        npy_path,
        labels,
        train_indices,
        mean,
        std,
        crop_size=crop_size,
        crop_stride=crop_stride,
        is_training=True,
    )
    
    val_dataset = PTBXLECGDataset(
        npy_path,
        labels,
        val_indices,
        mean,
        std,
        crop_size=crop_size,
        crop_stride=crop_stride,
        is_training=False,
    )
    
    test_dataset = PTBXLECGDataset(
        npy_path,
        labels,
        test_indices,
        mean,
        std,
        crop_size=crop_size,
        crop_stride=crop_stride,
        is_training=False,
    )
    
    # Load config and model
    config = ViTNepaECGConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_classes,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.is_causal = model_args.is_causal
    
    model = ViTNepaECGForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Set loss function for multi-label classification
    def loss_function(labels, logits, config=None, **kwargs):
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    
    model.loss_function = loss_function
    
    # Collate functions
    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        labels = torch.stack([e["labels"] for e in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Initialize trainer
    trainer = ECGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_auc_metrics,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir):
            checkpoint = get_last_checkpoint(training_args.output_dir)
        
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
    
    # Test evaluation
    if training_args.do_predict:
        test_results = trainer.predict(test_dataset)
        logger.info(f"Test AUC: {test_results.metrics.get('test_auc', 'N/A')}")


if __name__ == "__main__":
    main()

