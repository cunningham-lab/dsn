#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=LRRNN_DSN
#SBATCH -c 1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=10gb

source activate dsn
python3 train_rank2_CDD.py $1 $2 $3 $4
