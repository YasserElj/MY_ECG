#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-gpu
#SBATCH --partition gpu
#SBATCH --qos default-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=jepa_pretrain     # name of the job
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G

#SBATCH --output=outputs/pretrain.log
#SBATCH --error=errors/pretrain.log



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


python -m pretrain \
  --data "mimic-iv-ecg=../dataset/mimic-ecg.npy" \
  --out "checkpoints/" \
  --config "ViTS_mimic" \
  --amp "bfloat16"
