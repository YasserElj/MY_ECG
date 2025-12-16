#!/bin/bash
# MyJEPA Pretraining with 1x GPU
# Usage: ./batch/pretrain.sh

mkdir -p outputs checkpoints

# Activate conda
eval "$(conda shell.bash hook)"
conda activate jepa-ecg

# Run with nohup (survives disconnect)
nohup python pretrain.py \
    --data "mimic-iv-ecg=../dataset/Mimic-IV-All/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16" \
    --wandb \
    --entity "AtlasVision_CC" \
    --run-name "MyJEPA_pretrain" \
    --seed 42 \
    > outputs/pretrain.log 2>&1 &

echo "Training started! PID: $!"
echo "Monitor: tail -f outputs/pretrain.log"

