# MyJEPA: Single-Encoder JEPA with SIGReg

MyJEPA combines ECG-JEPA's masking + prediction with LeJEPA's single-encoder + SIGReg approach.

## Key Features

- **Single Encoder**: No EMA target encoder - same encoder for context and target
- **Masking Strategy**: ECG-JEPA style block masking (15-25% context, 75-85% targets)
- **Predictor**: Transformer-based predictor to predict target embeddings from context
- **SIGReg**: Sketched Isotropic Gaussian Regularization for collapse prevention

## Architecture

```
Input ECG (B, 12, 5000)
    │
    ├──── MaskCollator ────┐
    │                      │
    ▼                      ▼
Context Mask          Target Mask
    │                      │
    ▼                      ▼
Encoder(masked)       Encoder(full)
    │                      │
    ▼                      ▼
z_ctx (B, K_ctx, dim)  z_full (B, N, dim)
    │                      │
    │                      ├──── extract ──── z_tgt (B, K_tgt, dim)
    │                      │
    ▼                      ▼
Predictor ─────────► z_pred    Projector
    │                 │            │
    │                 │            ▼
    ▼                 ▼         SIGReg
Prediction Loss = |z_pred - z_tgt|    │
    │                                  │
    └─────────────── Total Loss = pred_loss + λ * sigreg_loss
```

## Loss Function

```
L = L_pred + λ * L_sigreg

where:
- L_pred = smooth_L1(z_pred, z_tgt)  # Prediction loss
- L_sigreg = SIGReg(Projector(mean(z_full)))  # Regularization loss
- λ = 0.02 (default)
```

## Project Structure

```
MyJEPA/
├── models/
│   ├── myjepa.py          # Main model class
│   ├── vit.py             # VisionTransformer encoder
│   ├── predictor.py       # Transformer predictor
│   ├── vit_classifier.py  # Classifier for finetuning
│   ├── modules.py         # Block, MLP, Projector, etc.
│   └── utils.py           # apply_mask, pos_embed
├── data/
│   ├── masks.py           # MaskCollator
│   ├── datasets/          # MIMIC-IV-ECG, PTB-XL
│   ├── transforms.py      # ECG transforms
│   └── utils.py           # Data loading
├── utils/
│   ├── monitoring.py      # AverageMeter, GPU stats
│   └── schedules.py       # LR schedules
├── configs/
│   ├── ViTS_mimic.yaml    # Pretraining config
│   └── eval/linear.yaml   # Finetuning config
├── batch/
│   ├── pretrain.sh        # Single-GPU pretraining
│   ├── pretrain_2gpu.sh   # Multi-GPU pretraining
│   └── finetune/*.sh      # Finetuning scripts
├── pretrain.py            # Pretraining script
├── finetune.py            # Finetuning script
└── logging.ini            # Logging config
```

## Usage

### Pretraining

Single GPU:
```bash
python pretrain.py \
    --data "mimic-iv-ecg=path/to/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16" \
    --wandb \
    --seed 42
```

Multi-GPU (2 GPUs):
```bash
torchrun --standalone --nproc_per_node=2 pretrain.py \
    --data "mimic-iv-ecg=path/to/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16" \
    --wandb \
    --seed 42
```

### Finetuning

```bash
python finetune.py \
    --data-dir "path/to/ptb-xl" \
    --dump "path/to/ptb-xl.npy" \
    --encoder "checkpoints/best_ckpt.pt" \
    --task "rhythm" \
    --wandb
```

### SLURM

```bash
# Pretraining
sbatch batch/pretrain.sh

# Finetuning
sbatch batch/finetune/rhythm.sh
```

## Configuration

### Pretraining Config (ViTS_mimic.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| dim | 384 | Encoder dimension |
| depth | 8 | Number of transformer blocks |
| num_heads | 6 | Attention heads |
| pred_dim | 192 | Predictor dimension |
| pred_depth | 8 | Predictor transformer blocks |
| sigreg_lambda | 0.02 | SIGReg weight |
| min_keep_ratio | 0.15 | Minimum context ratio |
| max_keep_ratio | 0.25 | Maximum context ratio |

## Key Differences from ECG-JEPA

| Aspect | ECG-JEPA | MyJEPA |
|--------|----------|--------|
| Encoders | Context + Target (EMA) | Single encoder |
| Collapse prevention | EMA momentum | SIGReg regularization |
| Target gradient | stop_gradient | Normal gradient flow |
| Hyperparameters | EMA momentum schedule | SIGReg λ |

## Expected Performance

Target performance on PTB-XL rhythm classification:
- **0.92-0.95 AUC** (between LeJEPA 0.86 and ECG-JEPA 0.96)

## wandb Metrics

Pretraining:
- `train/loss`, `train/pred_loss`, `train/sigreg_loss`
- `train/lr`, `train/step_time`, `train/throughput`
- `gradients/norm`
- `system/gpu_memory_*`

Finetuning:
- `train/loss`, `train/lr`
- `val/auc`, `val/best_auc`
- `test/auc`, per-class AUCs

