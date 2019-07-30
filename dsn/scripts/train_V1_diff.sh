#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=V1diff
#SBATCH -c 1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=2gb

source activate dsn
python3 train_V1_diff.py $1 $2 $3 $4 $5
