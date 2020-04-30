#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=LRRNN_DSN
#SBATCH -c 1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=2gb

source activate dsn
python3 train_LowRankRNN.py $1 $2 $3 $4
