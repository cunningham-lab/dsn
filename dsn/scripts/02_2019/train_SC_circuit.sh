#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=SC_DSN
#SBATCH -c 1
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=5gb

source activate dsn
python3 train_SC_circuit.py $1 $2 $3 $4 $5
