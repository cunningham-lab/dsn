#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=rnn_dsn
#SBATCH -c 1
#SBATCH --time=30:30:00
#SBATCH --mem-per-cpu=2gb

source activate dsn
python3 test_V1_circuit.py $1 $2 $3 $4
