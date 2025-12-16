#!/bin/bash
#SBATCH --job-name=myjepa-pretrain-2gpu
#SBATCH --output=outputs/pretrain_2gpu_%j.log
#SBATCH --error=errors/pretrain_2gpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# NCCL settings for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p outputs errors checkpoints

# Run multi-GPU pretraining with torchrun
torchrun --standalone --nproc_per_node=2 pretrain.py \
    --data "mimic-iv-ecg=../dataset/Mimic-IV-All/mimic-ecg-mini.npy" \
    --out "checkpoints/" \
    --config "ViTS_mimic" \
    --amp "bfloat16" \
    --wandb \
    --entity "AtlasVision_CC" \
    --run-name "MyJEPA_2gpu_pretrain" \
    --seed 42

echo "2-GPU pretraining completed!"

