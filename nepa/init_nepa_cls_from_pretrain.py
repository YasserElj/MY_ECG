import os
import argparse
import torch
import torch.nn as nn

from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download

from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
from models.vit_nepa.modeling_vit_nepa import (
    ViTNepaForPreTraining,
    ViTNepaForImageClassification,
)


def fold_layerscale_into_dense_if_needed(
    cfg: ViTNepaConfig, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Fold LayerScale parameters into dense layers if config disables LayerScale."""
    if getattr(cfg, "layerscale_value", None) is not None:
        return state_dict

    ls_keys = [k for k in state_dict.keys() if k.endswith("layer_scale.lambda1")]
    if not ls_keys:
        return state_dict

    print(
        f"Detected {len(ls_keys)} LayerScale parameters in checkpoint "
        "while config.layerscale_value is None. Folding into dense layers..."
    )

    new_sd = dict(state_dict)

    for ls_key in ls_keys:
        lambda1 = new_sd[ls_key]
        if not isinstance(lambda1, torch.Tensor):
            continue

        if ".output.layer_scale.lambda1" in ls_key:
            base = ls_key.replace(".layer_scale.lambda1", "")
            dense_prefix = base + ".dense"
        else:
            base = ls_key.replace(".layer_scale.lambda1", "")
            dense_prefix = base + ".attention.output.dense"

        w_key = dense_prefix + ".weight"
        b_key = dense_prefix + ".bias"

        if w_key not in new_sd:
            print(
                f"Skip folding for {ls_key}: dense weight '{w_key}' not found in state_dict."
            )
            continue

        W = new_sd[w_key]
        if W.shape[0] != lambda1.shape[0]:
            print(
                f"Shape mismatch when folding {ls_key}: "
                f"lambda1 shape {tuple(lambda1.shape)}, weight shape {tuple(W.shape)}, skip."
            )
            continue

        lambda_expanded = lambda1.unsqueeze(1)
        new_sd[w_key] = W * lambda_expanded

        if b_key in new_sd and new_sd[b_key] is not None:
            new_sd[b_key] = new_sd[b_key] * lambda1

        del new_sd[ls_key]

    print("LayerScale folding finished.")
    return new_sd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ViTNepaForImageClassification from ViTNepaForPreTraining."
    )

    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        required=True,
        help="HF repo id of ViTNepaForPreTraining, e.g. SixAILab/nepa-patch14-gelu-l",
    )
    parser.add_argument(
        "--pretrained_revision",
        type=str,
        default="main",
        help="Revision (branch/tag/commit) for the pretrained model.",
    )

    parser.add_argument(
        "--config_model_id",
        type=str,
        default=None,
        help="HF repo id to load ViTNepaConfig from. "
             "If None, use --pretrained_model_id.",
    )
    parser.add_argument(
        "--config_revision",
        type=str,
        default=None,
        help="Revision for config repo. If None, use --pretrained_revision.",
    )

    parser.add_argument(
        "--new_model_id",
        type=str,
        default=None,
        help="HF repo id to push the new classification model to.",
    )
    parser.add_argument(
        "--new_revision",
        type=str,
        default="init",
        help="Revision (branch/tag) for the new model on Hub.",
    )

    parser.add_argument(
        "--num_labels",
        type=int,
        default=1000,
        help="Number of labels for the classification head.",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HF token. If not provided, will use HF_TOKEN env variable.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, push new model as private repo on Hub.",
    )

    parser.add_argument(
        "--save_local",
        action="store_true",
        help="If set, save model/config/processor to local_dir.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory to save the new model. "
             "If None and save_local is set, will be derived from new_model_id.",
    )

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, push the new model to Hugging Face Hub.",
    )

    parser.add_argument(
        "--disable_layerscale",
        action="store_true",
        help="If set, set config.layerscale_value=None and fold LayerScale into dense.",
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="If set, load backbone from pytorch_model_ema.bin instead of default weights.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_model_id is None:
        args.config_model_id = args.pretrained_model_id
    if args.config_revision is None:
        args.config_revision = args.pretrained_revision

    if args.push_to_hub and args.new_model_id is None:
        raise ValueError(
            "--push_to_hub is set but --new_model_id is not provided."
        )

    if args.token is None and (args.push_to_hub or args.private):
        raise ValueError(
            "No HF token provided. Use --token or set HF_TOKEN environment variable."
        )

    print(
        f"Loading ViTNepaConfig from {args.config_model_id} "
        f"(revision={args.config_revision})..."
    )
    config = ViTNepaConfig.from_pretrained(
        args.config_model_id,
        revision=args.config_revision,
        token=args.token,
    )

    config.num_labels = args.num_labels
    if args.disable_layerscale:
        config.layerscale_value = None
    config.architectures = ["ViTNepaForImageClassification"]

    print(
        f"Loading image processor from {args.pretrained_model_id} "
        f"(revision={args.pretrained_revision})..."
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.pretrained_model_id,
        revision=args.pretrained_revision,
        token=args.token,
    )

    # Load backbone weights: EMA or standard
    if args.use_ema:
        print(
            f"Loading EMA checkpoint (pytorch_model_ema.bin) from {args.pretrained_model_id} "
            f"(revision={args.pretrained_revision})..."
        )
        ckpt_path = hf_hub_download(
            repo_id=args.pretrained_model_id,
            filename="pytorch_model_ema.bin",
            revision=args.pretrained_revision,
            token=args.token,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        backbone_sd = {}
        for k, v in ckpt.items():
            # Strip top-level "vit_nepa." prefix if present
            if k.startswith("vit_nepa."):
                backbone_sd[k[len("vit_nepa."):]] = v
            else:
                backbone_sd[k] = v
    else:
        print(
            f"Loading ViTNepaForPreTraining from {args.pretrained_model_id} "
            f"(revision={args.pretrained_revision})..."
        )
        pretrain_model = ViTNepaForPreTraining.from_pretrained(
            args.pretrained_model_id,
            revision=args.pretrained_revision,
            token=args.token,
            torch_dtype=torch.float32,
        )
        backbone_sd = pretrain_model.vit_nepa.state_dict()

    print("Initializing ViTNepaForImageClassification from config...")
    cls_model = ViTNepaForImageClassification(config)

    print("Copying backbone weights into classification model...")
    backbone_sd = fold_layerscale_into_dense_if_needed(config, backbone_sd)
    missing, unexpected = cls_model.vit_nepa.load_state_dict(
        backbone_sd, strict=False
    )
    print(
        "load_state_dict(missing_keys, unexpected_keys) =",
        len(missing),
        len(unexpected),
    )
    if missing:
        print("  Missing keys example:", missing[:10])
    if unexpected:
        print("  Unexpected keys example:", unexpected[:10])

    if isinstance(cls_model.classifier, nn.Linear):
        nn.init.zeros_(cls_model.classifier.weight)
        if cls_model.classifier.bias is not None:
            nn.init.zeros_(cls_model.classifier.bias)

    if args.save_local:
        if args.local_dir is None:
            if args.new_model_id is not None:
                args.local_dir = args.new_model_id.split("/")[-1]
            else:
                args.local_dir = "vitnepa_cls"
        os.makedirs(args.local_dir, exist_ok=True)
        print(f"Saving model/config/processor to {args.local_dir} ...")
        cls_model.save_pretrained(args.local_dir)
        config.save_pretrained(args.local_dir)
        image_processor.save_pretrained(args.local_dir)
        print("Local save done.")

    if args.push_to_hub:
        push_kwargs = {
            "revision": args.new_revision,
            "private": args.private,
            "token": args.token,
        }
        print(
            f"Pushing model to {args.new_model_id} "
            f"(revision={args.new_revision}, private={args.private})..."
        )
        cls_model.push_to_hub(args.new_model_id, **push_kwargs)
        config.push_to_hub(args.new_model_id, **push_kwargs)
        image_processor.push_to_hub(args.new_model_id, **push_kwargs)
        print("Push to Hub done.")


if __name__ == "__main__":
    main()
