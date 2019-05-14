#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=STG_DSN
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=100gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate dsn_gpu
python3 train_STG_circuit.py $1 $2 $3 $4 $5

