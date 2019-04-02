#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=r2CS_DSN
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=10gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate dsn_gpu
python3 train_rank2_CDD_static.py $1 $2 $3 $4
