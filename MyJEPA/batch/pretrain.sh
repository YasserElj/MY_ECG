#!/bin/bash
#SBATCH --job-name=myjepa-pretrain
#SBATCH --output=outputs/pretrain_%j.log
#SBATCH --error=errors/pretrain_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p outputs errors checkpoints

# Run pretraining
python pretrain.py \
    --data "mimic-iv-ecg=../dataset/Mimic-IV-All/mimic-ecg-mini.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16" \
    --wandb \
    --entity "AtlasVision_CC" \
    --run-name "MyJEPA_pretrain" \
    --seed 42

echo "Pretraining completed!"

