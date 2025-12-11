#!/bin/bash
# ================================================================
# LeJEPA-ECG Pretraining with 2x GPUs (DDP)
# For: 2x NVIDIA RTX 6000 Ada (48GB each)
# ================================================================
# Usage (local machine, no SLURM):
#   chmod +x batch/pretrain_2gpu.sh
#   ./batch/pretrain_2gpu.sh
# ================================================================

set -e

# Create output directories
mkdir -p outputs errors checkpoints

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "========================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate jepa-ecg

# Performance settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Run with 2 GPUs using torchrun
torchrun --standalone --nproc_per_node=2 pretrain.py \
    --data "mimic-iv-ecg=../dataset/Mimic-IV-All/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic_rtx6000" \
    --amp "bfloat16" \
    --wandb \
    --run-name "LeJEPA_2gpu_pretrain" \
    --seed 42 \
    2>&1 | tee outputs/pretrain_2gpu.log

echo "Training complete!"

