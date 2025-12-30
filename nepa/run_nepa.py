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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# import evaluate
import numpy as np
import torch
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
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.versions import require_version

from models.vit_nepa import ViTNepaForPreTraining, ViTNepaConfig

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class EnhancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        embed_lr=None,
        ema_decay=0.9999,
        use_ema=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embed_lr = embed_lr
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"layer_scale", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.embed_lr is None:
            return super().create_optimizer()

        decay_params = set(self.get_decay_parameter_names(self.model))
        embed_params = set(
            f"vit_nepa.embeddings.{n}"
            for n, _ in self.model.vit_nepa.embeddings.named_parameters()
        )

        wd = self.args.weight_decay
        base_lr = self.args.learning_rate

        groups = [
            {"params": [], "weight_decay": wd,  "lr": self.embed_lr},
            {"params": [], "weight_decay": 0.0, "lr": self.embed_lr},
            {"params": [], "weight_decay": wd,  "lr": base_lr},
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
    # label_column_name: str = field(
    #     default="label",
    #     metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    # )
    load_from_disk: bool = field(default=False, metadata={"help": "Load from disk"})
    keep_in_memory: bool = field(default=False, metadata={"help": "keep_in_memory"})

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
        default=None,
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
    embed_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for embedding layer parameters."},
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
    # if data_args.label_column_name not in dataset_column_names:
    #     raise ValueError(
    #         f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
    #         "Make sure to set `--label_column_name` to the correct text column - one of "
    #         f"{', '.join(dataset_column_names)}."
    #     )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        return {"pixel_values": pixel_values}
        # labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        # return {"pixel_values": pixel_values, "labels": labels}

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    # labels = dataset["train"].features[data_args.label_column_name].names
    # label2id, id2label = {}, {}
    # for i, label in enumerate(labels):
    #     label2id[label] = str(i)
    #     id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    # metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    # def compute_metrics(p):
    #     """Computes accuracy on a batch of predictions"""
    #     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = ViTNepaConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        # num_labels=len(labels),
        # label2id=label2id,
        # id2label=id2label,
        # finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.model_name_or_path:
        model = ViTNepaForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        logger.info("Training new model from scratch")
        model = ViTNepaForPreTraining(config)
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )

    # Define torchvision transforms to be applied to each image.
    if isinstance(image_processor, TimmWrapperImageProcessor):
        # Pretraining: only random crop + random flip + normalize
        _train_transforms = Compose([
            RandomResizedCrop(image_processor.input_size, interpolation=3),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.mean, std=image_processor.std),
        ])
        # Validation: resize + center crop + normalize
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
        _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        _val_transforms = Compose(
            [
                Resize(size),
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

    # Initialize our trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        # compute_metrics=compute_metrics,
        processing_class=image_processor,
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
    # if training_args.do_eval:
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "embedded-prediction",
        "dataset": data_args.dataset_name,
        "tags": ["embedded-prediction", "vision"],
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