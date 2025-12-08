#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lejepa_finetune_form
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

#SBATCH --output=outputs/finetune_form.log
#SBATCH --error=errors/finetune_form.log

module purge
mkdir -p outputs errors eval

eval "$(conda shell.bash hook)"
conda activate jepa-ecg

echo "=== GPU Information ==="
nvidia-smi
echo "========================"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python -m finetune \
    --data-dir "../../ecg/data/ptb-xl" \
    --encoder "checkpoints/best_ckpt.pt" \
    --out "eval/form" \
    --config "linear" \
    --task "form" \
    --wandb \
    --run-name "finetune_form"

