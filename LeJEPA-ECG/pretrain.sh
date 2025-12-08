#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=512_lejepa_ecg_pretrain
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

#SBATCH --output=outputs/pretrain_bs512.log
#SBATCH --error=errors/pretrain_bs512.log

module purge
mkdir -p outputs errors checkpoints512

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
    --out "checkpoints512/" \
    --config "ViTS_mimic_a100" \
    --amp "bfloat16" \
    --wandb \
    --run-name "pretrain_bs512" \
    --seed 42
    # --resume "checkpoints/chkpt_30000.pt"

