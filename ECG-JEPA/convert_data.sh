#!/bin/bash
#SBATCH --account recsys-y8xnv0gztug-default-cpu
#SBATCH --partition compute
#SBATCH --qos default-cpu
#SBATCH --job-name=convert     # name of the job

#SBATCH --output=outputs/convert.log
#SBATCH --error=errors/convert.log

#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --open-mode=append



module purge
mkdir -p outputs errors

# Activate the Surformer conda environment
eval "$(conda shell.bash hook)"
conda activate jepa-ecg

# Let BLAS/NumExpr use the cores you requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1   # live progress/logs


python -m scripts.dump_data --data-dir "../../ecg/data/mimic-ecg" --dataset "mimic-iv-ecg" --verbose
echo "Data conversion completed."