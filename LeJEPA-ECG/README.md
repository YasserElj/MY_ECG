# LeJEPA-ECG

Self-supervised ECG representation learning using **LeJEPA** (Lean Joint-Embedding Predictive Architecture).

## Overview

LeJEPA-ECG adapts the LeJEPA framework for ECG signals. Unlike ECG-JEPA which uses masking and prediction, LeJEPA uses:

- **Multi-view augmentation**: Multiple augmented views of each ECG
- **Invariance loss**: Different views should have similar embeddings
- **SIGReg regularization**: Embeddings follow isotropic Gaussian distribution

Key advantages:
- No teacher-student architecture (no EMA)
- No masking or patch prediction
- Single hyperparameter (λ)
- Simpler, more stable training

## Architecture

```
ECG Signal (12 leads, 5000 samples)
    │
    ├── Augmentation ──► View 1 ──┐
    ├── Augmentation ──► View 2 ──┤
    ├── Augmentation ──► View 3 ──┼──► ViT Encoder ──► Projector ──► Embeddings
    └── Augmentation ──► View 4 ──┘
                                                            │
                                                            ▼
                                              Loss = λ*SIGReg + (1-λ)*Invariance
```

## Usage

### Pretraining

```bash
python pretrain.py \
    --data "mimic-iv-ecg=path/to/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16"
```

### Configuration

Edit `configs/ViTS_mimic.yaml`:

```yaml
# LeJEPA parameters
num_views: 4        # Number of augmented views
lambda: 0.02        # SIGReg weight (invariance = 1-lambda)
proj_dim: 128       # Projector output dimension
num_slices: 1024    # Slices for SIGReg

# Augmentation
crop_scale: [0.5, 1.0]
amplitude_scale: [0.8, 1.2]
noise_std: 0.05
```

## Key Differences from ECG-JEPA

| Aspect | ECG-JEPA | LeJEPA-ECG |
|--------|----------|------------|
| Strategy | Masking + Prediction | Multi-view + Invariance |
| Architecture | Teacher-Student (EMA) | Single encoder |
| Loss | Reconstruction (L1) | Invariance + SIGReg |
| Complexity | Momentum scheduling, predictor | Single λ parameter |

## ECG Augmentations

- **RandomCrop**: Different time windows (50-100% of signal)
- **AmplitudeScale**: Random amplitude scaling (0.8-1.2x)
- **GaussianNoise**: Additive noise (σ=0.05)
- **TimeShift**: Circular temporal shift
- **BaselineWander**: Simulated low-frequency drift

## References

- [LeJEPA Paper](https://arxiv.org/abs/2511.08544) - Balestriero & LeCun, 2025
- [ECG-JEPA Paper](https://arxiv.org/abs/2410.20514) - Chen et al., 2024

