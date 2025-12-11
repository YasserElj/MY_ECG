#!/bin/bash
# LeJEPA-ECG Pretraining with 2x GPUs (DDP)
# Usage: ./batch/pretrain_2gpu.sh

mkdir -p outputs checkpoints

# Activate conda
eval "$(conda shell.bash hook)"
conda activate jepa-ecg

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# Run with nohup (survives disconnect)
nohup torchrun --standalone --nproc_per_node=2 pretrain.py \
    --data "mimic-iv-ecg=../dataset/Mimic-IV-All/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic_rtx6000" \
    --amp "bfloat16" \
    --wandb \
    --run-name "LeJEPA_2gpu_pretrain" \
    --seed 42 \
    > outputs/pretrain_2gpu.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor: tail -f outputs/pretrain_2gpu.log"
