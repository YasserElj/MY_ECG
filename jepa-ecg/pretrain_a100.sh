#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fact_resume
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --output=outputs/resume_factorized.log
#SBATCH --error=errors/resume_factorized.log

module purge
mkdir -p outputs errors

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate jepa-ecg

echo "=== GPU Information ==="
nvidia-smi
echo "========================"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Resume Training Command
# Note: "pretrain-output-st" should be the same directory you used before
# to ensure it finds the previous logs/checkpoints if needed.

python -m pretrain \
  --data "mimic-iv-ecg=../dataset/mimic-ecg-small.npy" \
  --out "pretrain-output-st" \
  --config "ViTS_mimic_A100" \
  --chkpt "pretrain-output-st/chkpt_35000.pt" \
  --amp "bfloat16" \
  --wandb \
  --wandb-entity "AtlasVision_CC" \
  --wandb-project "Physio-JEPA-ECG" \
  --run-name "Physio-JEPA_ST_resume_35k"