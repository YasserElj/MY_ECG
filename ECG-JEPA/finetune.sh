#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ecg_finetune     # name of the job
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

#SBATCH --output=outputs/finetune4_rhythm.log
#SBATCH --error=errors/finetune4_rhythm.log



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


# python -m finetune \
#   --data-dir "../../ecg/data/ptb-xl" \
#   --encoder "checkpoints2/best_ckpt.pt" \
#   --out "checkpoints_finetune_best" \
#   --config "linear" \
#   --task "all"

# python -m finetune \
#   --data-dir "../../ecg/data/ptb-xl" \
#   --encoder "checkpoints4/chkpt_100000.pt" \
#   --out "checkpoints_finetune_4" \
#   --config "linear" \
#   --task "all"


# rhythm

python -m finetune \
  --data-dir "../../ecg/data/ptb-xl" \
  --encoder "checkpoints4/best_ckpt.pt" \
  --out "checkpoints_finetune_rhythm" \
  --config "linear" \
  --task "rhythm"

# superdiagnostic

# python -m finetune \
#   --data-dir "../../ecg/data/ptb-xl" \
#   --encoder "checkpoints2/chkpt_100000.pt" \
#   --out "checkpoints_finetune_superdiagnostic" \
#   --config "linear" \
#   --task "superdiagnostic"
