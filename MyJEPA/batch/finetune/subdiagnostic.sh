#!/bin/bash
#SBATCH --job-name=myjepa-subdiag
#SBATCH --output=outputs/finetune_subdiag_%j.log
#SBATCH --error=errors/finetune_subdiag_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd $SLURM_SUBMIT_DIR

mkdir -p outputs errors eval

python finetune.py \
    --data-dir "../dataset/PTB-XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" \
    --dump "../dataset/PTB-XL/ptb-xl.npy" \
    --encoder "checkpoints/best_ckpt.pt" \
    --out "eval" \
    --config "linear" \
    --task "subdiagnostic" \
    --amp "bfloat16" \
    --wandb \
    --entity "AtlasVision_CC" \
    --seed 42

echo "Subdiagnostic finetuning completed!"

