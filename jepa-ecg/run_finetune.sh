#!/bin/bash

# 1. Configuration
export CUDA_VISIBLE_DEVICES=1

# --- USE ABSOLUTE PATHS TO PREVENT ERRORS ---
# Run `pwd` in your terminal to find your current path and fill these in:
DATA_DIR="../dataset/ptb-xl"  # <--- Verify this path contains scp_statements.csv
PRETRAINED_CHKPT="pretrain-output-st/chkpt_100000.pt"


# Settings
TASK="all"
TASK_RUN_NAME="${TASK}_2"
WANDB_ENTITY="AtlasVision_CC"
WANDB_PROJECT="Physio-JEPA-Finetune"

# Output Dirs
OUT_LINEAR="ft-output-st/${TASK_RUN_NAME}/linear"
OUT_FINETUNE="ft-output-st/${TASK_RUN_NAME}/finetune_standard"
OUT_AFTER_LINEAR="ft-output-st/${TASK_RUN_NAME}/finetune_after_linear"

# 2. Create directories
mkdir -p "$OUT_LINEAR"
mkdir -p "$OUT_FINETUNE"
mkdir -p "$OUT_AFTER_LINEAR"

CMD="
# --- STEP 1: Linear Evaluation (Probing) ---
# Freezes encoder, trains only the head.
echo '[1/3] Starting Linear Evaluation...';
python -m finetune \
  --data-dir \"${DATA_DIR}\" \
  --encoder \"${PRETRAINED_CHKPT}\" \
  --out \"${OUT_LINEAR}\" \
  --config \"linear\" \
  --task \"${TASK}\" \
  --amp \"bfloat16\" \
  --wandb \
  --wandb-entity \"${WANDB_ENTITY}\" \
  --wandb-project \"${WANDB_PROJECT}\" \
  --run-name \"Linear-Probe-${TASK_RUN_NAME}\";

# --- STEP 2: Standard Full Fine-Tuning ---
# Unfreezes encoder immediately. Starts from PRE-TRAINED weights.
echo '[2/3] Starting Standard Fine-tuning...';
python -m finetune \
  --data-dir \"${DATA_DIR}\" \
  --encoder \"${PRETRAINED_CHKPT}\" \
  --out \"${OUT_FINETUNE}\" \
  --config \"finetune\" \
  --task \"${TASK}\" \
  --amp \"bfloat16\" \
  --wandb \
  --wandb-entity \"${WANDB_ENTITY}\" \
  --wandb-project \"${WANDB_PROJECT}\" \
  --run-name \"Finetune-Standard-${TASK_RUN_NAME}\";

# --- STEP 3: Fine-Tuning AFTER Linear (Optional) ---
# Loads the BEST checkpoint from Step 1 and fine-tunes it with low LR.
# This only runs if Step 1 succeeds.
if [ -f \"${OUT_LINEAR}/${TASK}_best_chkpt.pt\" ]; then
    echo '[3/3] Starting Fine-tuning After Linear...';
    python -m finetune \
      --data-dir \"${DATA_DIR}\" \
      --encoder \"${OUT_LINEAR}/${TASK}_best_chkpt.pt\" \
      --out \"${OUT_AFTER_LINEAR}\" \
      --config \"finetune_after_linear\" \
      --task \"${TASK}\" \
      --amp \"bfloat16\" \
      --wandb \
      --wandb-entity \"${WANDB_ENTITY}\" \
      --wandb-project \"${WANDB_PROJECT}\" \
      --run-name \"Finetune-After-Linear-${TASK_RUN_NAME}\";
else
    echo 'Skipping Step 3: Linear checkpoint not found.';
fi
"

# 3. Launch
nohup sh -c "$CMD" > "ft-output-st/${TASK_RUN_NAME}/finetune_phases_${TASK}.log" 2>&1 &

echo "Phased training started."
echo "Logs: ft-output-st/${TASK_RUN_NAME}/finetune_phases_${TASK}.log"
echo "PID: $!"
