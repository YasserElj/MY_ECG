# ECG Self-Supervised Learning

Self-supervised learning methods for ECG analysis: **ECG-JEPA** and **LeJEPA-ECG**.

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate jepa-ecg

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Datasets

Download from Hugging Face:

```bash
# MIMIC-IV-ECG (96GB) - for pretraining
huggingface-cli download Pryzl/MIMIIC_ECG_npy mimic-ecg.npy --local-dir ./dataset

# PTB-XL (2.6GB) - for finetuning
huggingface-cli download Pryzl/ptb-xl_npy ptb-xl.npy --local-dir ./dataset
```

Or download directly:
- [MIMIC-IV-ECG](https://huggingface.co/datasets/Pryzl/MIMIIC_ECG_npy)
- [PTB-XL](https://huggingface.co/datasets/Pryzl/ptb-xl_npy)

## Usage

### Pretraining

```bash
cd LeJEPA-ECG
python pretrain.py \
    --data "mimic-iv-ecg=../dataset/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic_a100" \
    --amp "bfloat16" \
    --wandb
```

### Finetuning

```bash
python finetune.py \
    --checkpoint "checkpoints/best_ckpt.pt" \
    --data "../dataset/ptb-xl.npy" \
    --task "diagnostic" \
    --wandb
```

## License

MIT
