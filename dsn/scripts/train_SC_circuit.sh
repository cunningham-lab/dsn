#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=SC_DSN
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --constraint=p100
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=5gb

module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5

source activate dsn_gpu
python3 train_SC_circuit.py $1 $2 $3
