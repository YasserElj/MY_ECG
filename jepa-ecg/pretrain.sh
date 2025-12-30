#!/bin/bash

# 1. Configuration
# Set visible devices (0,1 for the first two GPUs)
export CUDA_VISIBLE_DEVICES=0,1

# Paths
DATASET_PATH="../dataset/mimic-ecg.npy"
OUTPUT_DIR="pretrain-output"
CONFIG_NAME="ViTS_mimic"

# WandB Settings (Replace with your actual team name)
WANDB_ENTITY="AtlasVision_CC"
WANDB_PROJECT="Physio-JEPA-ECG"

# 2. Create output directory
mkdir -p "$OUTPUT_DIR"

# 3. Launch Command
# --nproc_per_node=2 tells torchrun to use 2 GPUs
# nohup keeps it running after you disconnect
nohup torchrun --nproc_per_node=2 pretrain.py \
  --data "mimic-iv-ecg=${DATASET_PATH}" \
  --out "${OUTPUT_DIR}" \
  --config "${CONFIG_NAME}" \
  --amp "bfloat16" \
  --wandb \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  > "${OUTPUT_DIR}/training_dist.log" 2>&1 &

# 4. Feedback
echo "Distributed pre-training started on 2 GPUs."
echo "Logs are being written to ${OUTPUT_DIR}/training_dist.log"
echo "Process ID (PID): $!"