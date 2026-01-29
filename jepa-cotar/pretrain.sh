#!/bin/bash
# 1. Configuration
# Set visible devices (0,1 for the first two GPUs)
export CUDA_VISIBLE_DEVICES=1

# Paths
DATASET_PATH="../dataset/mimic-ecg.npy"
OUTPUT_DIR="pretrain-output-st"
CONFIG_NAME="ViTS_mimic"
CHKPT_DIR="pretrain-output-st/chkpt_35000.pt"

# WandB Settings (Replace with your actual team name)
WANDB_ENTITY="AtlasVision_CC"
WANDB_PROJECT="Physio-JEPA-ECG"
WANDB_RUN="Physio-JEPA_ST_resume_35k"

# 2. Create output directory
mkdir -p "$OUTPUT_DIR"

# 3. Launch Command
# --nproc_per_node=2 tells torchrun to use 2 GPUs
# nohup keeps it running after you disconnect
nohup torchrun --nproc_per_node=1 pretrain.py \
  --data "mimic-iv-ecg=${DATASET_PATH}" \
  --out "${OUTPUT_DIR}" \
  --config "${CONFIG_NAME}" \
  --chkpt "${CHKPT_DIR}" \
  --amp "bfloat16" \
  --wandb \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --run-name "${WANDB_RUN}"\
  --seed 42 \
  > "${OUTPUT_DIR}/training_dist_resume_35k.log" 2>&1 &

# 4. Feedback
echo "Distributed pre-training started on 2 GPUs."
echo "Logs are being written to ${OUTPUT_DIR}/training_dist_resume_35k.log"
echo "Process ID (PID): $!"
