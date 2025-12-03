#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lejepa_ecg_pretrain
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

#SBATCH --output=outputs/pretrain.log
#SBATCH --error=errors/pretrain.log

module purge
mkdir -p outputs errors checkpoints

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate jepa-ecg

echo "=== GPU Information ==="
nvidia-smi
echo "========================"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python pretrain.py \
    --data "mimic-iv-ecg=../dataset/mimic-ecg.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16" \
    --wandb

