#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=r2CS_DSN
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --constraint=p100
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=5gb

source activate dsn_gpu
python3 train_rank2_CDD_static.py $1 $2 $3 $4
