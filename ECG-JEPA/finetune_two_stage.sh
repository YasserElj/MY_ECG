#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ecg_finetune_two_stage     # name of the job
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

#SBATCH --output=outputs/finetune_two_stage_rhythm.log
#SBATCH --error=errors/finetune_two_stage_rhythm.log



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
#   --encoder "checkpoints_finetune_4/all_best_chkpt.pt" \
#   --out "finetune-two-stage-all" \
#   --config "finetune_after_linear" \
#   --task "all"

# python -m finetune \
#   --data-dir "../../ecg/data/ptb-xl" \
#   --encoder "checkpoints4/chkpt_100000.pt" \
#   --out "checkpoints_finetune_4" \
#   --config "linear" \
#   --task "all"


# rhythm

# python -m finetune \
#   --data-dir "../../ecg/data/ptb-xl" \
#   --encoder "checkpoints2/chkpt_100000.pt" \
#   --out "checkpoints_finetune_rhythm" \
#   --config "linear" \
#   --task "rhythm"

python -m finetune \
  --data-dir "../../ecg/data/ptb-xl" \
  --encoder "checkpoints_finetune_rhythm/rhythm_best_chkpt.pt" \
  --out "finetune-two-stage-rhythm" \
  --config "finetune_after_linear" \
  --task "rhythm"

# superdiagnostic

# python -m finetune \
#   --data-dir "../../ecg/data/ptb-xl" \
#   --encoder "checkpoints2/chkpt_100000.pt" \
#   --out "checkpoints_finetune_superdiagnostic" \
#   --config "linear" \
#   --task "superdiagnostic"
