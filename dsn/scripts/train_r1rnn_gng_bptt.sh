#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=gng_bptt
#SBATCH -c 1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1gb

source activate dsn
python3 train_r1rnn_gng_bptt.py $1
