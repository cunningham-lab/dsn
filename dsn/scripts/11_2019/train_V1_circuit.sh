#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=V1f_DSN
#SBATCH -c 1
#SBATCH --time=15:59:00
#SBATCH --mem-per-cpu=2gb

source activate dsn
python3 train_V1_circuit.py $1 $2 $3 $4
