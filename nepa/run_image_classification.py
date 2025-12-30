# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file exceam in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "accelerate>=0.12.0",
#     "torch>=1.5.0",
#     "torchvision>=0.6.0",
#     "datasets>=2.14.0",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

import importlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, load_from_disk
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from timm.data import create_transform, Mixup

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoImageProcessor,
    HfArgumentParser,
    TimmWrapperImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, SchedulerType
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils.versions import require_version

from models.vit_nepa.modeling_vit_nepa import ViTNepaForImageClassification, ViTNepaConfig


""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def import_class(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


def soft_target_cross_entropy(labels, logits, config=None, **kwargs):
    smoothing = kwargs.get('smoothing', 0.0)
    
    logits = logits.float()
    labels = labels.float()
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    if labels.ndim == 1 or labels.size(-1) == 1:
        loss = F.nll_loss(log_probs, labels.long() if labels.ndim == 1 else labels.squeeze(-1).long(), reduction='none')
        return loss.mean()
    
    if smoothing > 0.0:
        soft_target_loss = -(labels * log_probs).sum(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        confidence = 1.0 - smoothing
        loss = confidence * soft_target_loss + smoothing * smooth_loss
        return loss.mean()
    else:
        return -(labels * log_probs).sum(dim=-1).mean()


class EnhancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_collator=None,
        ema_decay=0.999,
        use_ema=True,
        base_lr=1e-4,
        head_lr=1e-3,
        llrd=0.75,
        llrd_end=None,
        weight_decay=0.05,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None
        self.base_lr = base_lr
        self.head_lr = head_lr if head_lr is not None else base_lr
        self.llrd = llrd
        self.llrd_end = llrd_end
        self.weight_decay = weight_decay

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"layer_scale", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_collator is not None:
            old_collator = self.data_collator
            self.data_collator = self.eval_collator
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = old_collator
            return dataloader
        else:
            return super().get_eval_dataloader(eval_dataset)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        from collections import defaultdict

        base_lr = self.base_lr
        head_lr = self.head_lr
        llrd = self.llrd
        weight_decay = self.weight_decay

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)

        no_llrd = []

        head_param_ids = set()
        if hasattr(self.model, "classifier"):
            head_param_ids.update(id(p) for p in self.model.classifier.parameters())

        final_ln_ids = set()
        if hasattr(self.model, "vit_nepa") and hasattr(self.model.vit_nepa, "layernorm"):
            final_ln_ids.update(id(p) for p in self.model.vit_nepa.layernorm.parameters())

        embeddings_ids = set()
        if hasattr(self.model, "vit_nepa") and hasattr(self.model.vit_nepa, "embeddings"):
            embeddings_ids.update(id(p) for p in self.model.vit_nepa.embeddings.parameters())

        layers_param_ids = []
        mlp_layers_param_ids = []
        if hasattr(self.model, "vit_nepa") and hasattr(self.model.vit_nepa, "encoder") and hasattr(self.model.vit_nepa.encoder, "layer"):
            encoder_layers = list(self.model.vit_nepa.encoder.layer)
            for blk in encoder_layers:
                layers_param_ids.append(set(id(p) for p in blk.parameters()))
                mlp_set = set()
                if hasattr(blk, "intermediate"):
                    mlp_set.update(id(p) for p in blk.intermediate.parameters())
                if hasattr(blk, "output"):
                    mlp_set.update(id(p) for p in blk.output.parameters())
                mlp_layers_param_ids.append(mlp_set)
        else:
            encoder_layers = []
        num_layers = len(encoder_layers)

        grouped = defaultdict(list)

        for full_name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue

            if any(tag in full_name for tag in no_llrd):
                lr_base = base_lr
                scale = 0
            elif id(p) in head_param_ids or id(p) in final_ln_ids:
                lr_base = head_lr
                scale = 0
            else:
                assigned = False
                for i in range(num_layers):
                    if id(p) in layers_param_ids[i]:
                        lr_base = base_lr
                        scale = (num_layers - 1 - i)
                        assigned = True
                        break
                if not assigned:
                    if id(p) in embeddings_ids:
                        lr_base = base_lr
                        scale = num_layers
                    else:
                        lr_base = base_lr
                        scale = 0

            if p.ndim <= 1:
                wd = 0.0
            else:
                wd = weight_decay if full_name in decay_parameters else 0.0
            grouped[(lr_base, wd, scale)].append(p)

        optimizer_grouped_parameters = []
        for (lr_base, wd, scale), params in grouped.items():
            optimizer_grouped_parameters.append({
                "params": params,
                "lr": lr_base,
                "weight_decay": wd,
                "llrd": llrd,
                "llrd_scale": scale,
            })

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = self.optimizer if optimizer is None else optimizer
        scheduler_specific_kwargs = self.args.lr_scheduler_kwargs or {}
        sched_name = scheduler_specific_kwargs.pop("custom_scheduler_type", None)

        if sched_name == "llrd_cosine":
            from schedulers import get_llrd_cosine_schedule

            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
            self.lr_scheduler = get_llrd_cosine_schedule(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **scheduler_specific_kwargs,
            )
            self._created_lr_scheduler = True

        if sched_name == "llrd_cosine_warmup":
            from schedulers import get_llrd_cosine_schedule_with_warmup

            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
            self.lr_scheduler = get_llrd_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **scheduler_specific_kwargs,
            )
            self._created_lr_scheduler = True

        # HF will only create a scheduler if self.lr_scheduler is still None
        super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def _init_ema_model(self):
        if self.ema_model is None:
            import copy
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            self.ema_model = self.ema_model.float()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_model is None:
            self._init_ema_model()
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    model_param = msd[k].float()
                    v.mul_(self.ema_decay).add_(model_param, alpha=1.0 - self.ema_decay)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        loss = soft_target_cross_entropy(labels, logits, smoothing=self.args.label_smoothing_factor if self.args.label_smoothing_factor else 0.0)

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.state.global_step > getattr(self, "_ema_global_step", 0):
            self._update_ema()
            self._ema_global_step = self.state.global_step
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix + "_ema", **gen_kwargs)
            self.model = backup

            out.update(ema_out)
        return out

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test", **gen_kwargs):
        out = super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

        if self.use_ema and self.ema_model is not None:
            backup = self.model
            self.model = self.ema_model
            ema_out = super().predict(test_dataset, ignore_keys, metric_key_prefix + "_ema", **gen_kwargs)
            self.model = backup

            from transformers.trainer_utils import PredictionOutput
            if isinstance(out, PredictionOutput) and isinstance(ema_out, PredictionOutput):
                out.metrics.update({k: v for k, v in ema_out.metrics.items()})
            else:
                out.update(ema_out)
        return out

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.use_ema and self.ema_model is not None and self.args.should_save:
            ema_path = f"{output_dir}/pytorch_model_ema.bin"
            os.makedirs(os.path.dirname(ema_path), exist_ok=True)
            torch.save(self.ema_model.state_dict(), ema_path)
            self.log({"ema_model_saved": ema_path})

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)

        if self.use_ema:
            ema_ckpt = os.path.join(resume_from_checkpoint, "pytorch_model_ema.bin")
            if os.path.exists(ema_ckpt):
                if self.ema_model is None:
                    import copy
                    self.ema_model = copy.deepcopy(self.model)
                    for p in self.ema_model.parameters():
                        p.requires_grad_(False)
                state_dict = torch.load(ema_ckpt, map_location="cpu")
                missing, unexpected = self.ema_model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    print(f"[EMA] Missing keys: {missing}, Unexpected keys: {unexpected}")
            else:
                print(f"[EMA] No EMA checkpoint found at {ema_ckpt}, starting fresh EMA.")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )
    load_from_disk: bool = field(default=False, metadata={"help": "Load from disk"})
    keep_in_memory: bool = field(default=False, metadata={"help": "keep_in_memory"})
    head_lr: float = field(default=1e-3, metadata={"help": "learning rate for classification head."})
    resize_size: Optional[int] = field(
        default=256, metadata={"help": "Resize size for input image."}
    )
    color_jitter: float = field(
        default=0.4, metadata={"help": "Color jitter factor (default: 0.4). "}
    )
    aa: Optional[str] = field(
        default="rand-m9-mstd0.5-inc1", metadata={"help": "Use AutoAugment policy. See timm for available policies."}
    )
    smoothing: float = field(
        default=0.1, metadata={"help": "Label smoothing factor."}
    )
    reprob: float = field(
        default=0.25, metadata={"help": "Random erase prob."}
    )
    remode: str = field(
        default="pixel", metadata={"help": "Random erase mode."}
    )
    recount: int = field(
        default=1, metadata={"help": "Random erase count."}
    )
    mixup: float = field(
        default=0.8, metadata={"help": "Mixup alpha."}
    )
    cutmix: float = field(
        default=1.0, metadata={"help": "Cutmix alpha."}
    )
    mixup_prob: float = field(
        default=1.0, metadata={"help": "Mixup probability."}
    )
    mixup_switch_prob: float = field(
        default=0.5, metadata={"help": "Mixup switch probability."}
    )
    mixup_mode: str = field(
        default="batch", metadata={"help": "Mixup mode."}
    )
    crop_pct: Optional[float] = field(
        default=None, metadata={"help": "Input image crop pct."}
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="SixAILab/nepa-large-patch14-224-sft",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    freeze_vit : bool = field(
        default=False,
        metadata={"help": "Whether to freeze vit backbone."},
    )
    freeze_vit_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of ViT layers to freeze."},
    )
    freeze_embed : bool = field(
        default=False,
        metadata={"help": "Whether to freeze patch embedding layer."},
    )
    is_causal: bool = field(
        default=False,
        metadata={"help": "Whether the model is causal model."},
    )
    hidden_dropout_prob: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout probability for fully connected layers in the embeddings and pooler."},
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout ratio for the attention probabilities."},
    )
    drop_path: Optional[float] = field(
        default=None,
        metadata={"help": "Drop path rate."},
    )
    use_ema: bool = field(
        default=True,
        metadata={"help": "Whether to use EMA model."},
    )
    ema_decay: float = field(
        default=0.9999,
        metadata={"help": "EMA decay."},
    )
    llrd: float = field(
        default=0.75,
        metadata={"help": "Layer-wise learning rate decay."},
    )
    llrd_end: Optional[float] = field(
        default=None,
        metadata={"help": "Layer-wise learning rate decay at the end of training."},
    )
    optimizer_cls: Optional[str] = field(
        default=None,
        metadata={"help": "Class of the optimizer like 'bitsandbytes.optim.LAMB'"},
    )
    optimizer_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "kwargs of the optimizer"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
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
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        if data_args.load_from_disk:
            dataset = load_from_disk(data_args.dataset_name, keep_in_memory=data_args.keep_in_memory)
        else:
            dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features[data_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    mixup_fn = Mixup(
        mixup_alpha=data_args.mixup,
        cutmix_alpha=data_args.cutmix,
        prob=data_args.mixup_prob,
        switch_prob=data_args.mixup_switch_prob,
        mode=data_args.mixup_mode,
        label_smoothing=data_args.smoothing,
        num_classes=len(labels),
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])

        if mixup_fn is not None:
            pixel_values, labels = mixup_fn(pixel_values, labels)

        return {"pixel_values": pixel_values, "labels": labels}

    def eval_collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir, keep_in_memory=True)

    # Define our compute_metrics function
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = ViTNepaConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.is_causal = model_args.is_causal
    if model_args.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if model_args.attention_probs_dropout_prob is not None:
        config.attention_probs_dropout_prob = model_args.attention_probs_dropout_prob
    if model_args.drop_path is not None:
        config.drop_path_prob = model_args.drop_path
    model = ViTNepaForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model.accepts_loss_kwargs = False
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )

    model.loss_function = soft_target_cross_entropy
    if model_args.freeze_vit:
        if model_args.freeze_vit_layers is None:
            for param in model.vit_nepa.parameters():
                param.requires_grad = False
            logger.info("Froze ViT backbone")
        else:
            total_layers = len(model.vit_nepa.encoder.layer)
            k = max(0, min(int(model_args.freeze_vit_layers), total_layers))
            if k != model_args.freeze_vit_layers:
                logger.warning(f"freeze_vit_layers clipped from {model_args.freeze_vit_layers} to {k} (total={total_layers})")

            for i in range(k):
                layer = model.vit_nepa.encoder.layer[i]
                for p in layer.parameters():
                    p.requires_grad = False
                logger.info(f"Froze ViT encoder layer {i}")

            logger.info(f"Froze first {k}/{total_layers} ViT encoder layers")

    if model_args.freeze_embed:
        for param in model.vit_nepa.embeddings.patch_embeddings.parameters():
            param.requires_grad = False
        logger.info("Froze Embedding Layer")

    if data_args.crop_pct is not None:
        if isinstance(image_processor, TimmWrapperImageProcessor):
            input_size = image_processor.input_size
        else:
            if "shortest_edge" in image_processor.size:
                input_size = image_processor.size["shortest_edge"]
            else:
                input_size = (image_processor.size["height"], image_processor.size["width"])
        if isinstance(input_size, (tuple, list)):
            size = tuple(int(s / data_args.crop_pct) for s in input_size)
        else:
            size = int(input_size / data_args.crop_pct)
        logger.info(f"Overriding resize_size to {size} from crop_pct {data_args.crop_pct}")
        data_args.resize_size = size

    # Define torchvision transforms to be applied to each image.
    if isinstance(image_processor, TimmWrapperImageProcessor):
        _train_transforms = create_transform(
            input_size=image_processor.input_size,
            is_training=True,
            color_jitter=data_args.color_jitter,
            auto_augment=data_args.aa,
            interpolation="bicubic",
            re_prob=data_args.reprob,
            re_mode=data_args.remode,
            re_count=data_args.recount,
            mean=image_processor.mean,
            std=image_processor.std,
        )
        _val_transforms = Compose([
            Resize(data_args.resize_size, interpolation=3),
            CenterCrop(image_processor.input_size),
            ToTensor(),
            Normalize(mean=image_processor.mean, std=image_processor.std),
        ])
    else:
        if "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
        else:
            size = (image_processor.size["height"], image_processor.size["width"])

        # Create normalization transform
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
            normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        else:
            normalize = Lambda(lambda x: x)
        # RandAugment(m=9, mstd=0.5) + RandomResizedCrop + Flip
        _train_transforms = create_transform(
            input_size=size,
            is_training=True,
            auto_augment=data_args.aa,
            interpolation="bicubic",
            re_prob=data_args.reprob,
            re_mode=data_args.remode,
            re_count=data_args.recount,
            mean=image_processor.image_mean if hasattr(image_processor, "image_mean") else None,
            std=image_processor.image_std if hasattr(image_processor, "image_std") else None,
        )
        _val_transforms = Compose(
            [
                Resize(data_args.resize_size, interpolation=3),  # bicubic
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    opt_tuple = None
    if model_args.optimizer_cls:
        opt_cls = import_class(model_args.optimizer_cls)
        kw = json.loads(model_args.optimizer_kwargs) if model_args.optimizer_kwargs else {}
        kw.setdefault("lr", getattr(training_args, "learning_rate", None) or 1e-3)
        opt_tuple = (opt_cls, kw)

    # Initialize our trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        optimizer_cls_and_kwargs=opt_tuple,
        compute_metrics=compute_metrics,
        processing_class=image_processor,
        data_collator=collate_fn,
        eval_collator=eval_collate_fn,
        ema_decay=model_args.ema_decay,
        use_ema=model_args.use_ema,
        base_lr=training_args.learning_rate,
        head_lr=data_args.head_lr,
        llrd=model_args.llrd,
        llrd_end=model_args.llrd_end,
        weight_decay=training_args.weight_decay,
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

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
