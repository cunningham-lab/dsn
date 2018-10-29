#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=linear_2D
#SBATCH -c 1
#SBATCH --time=11:30:00
#SBATCH --mem-per-cpu=2gb

source activate dsn
python3 test_linear_2D.py $1 $2 $3 $4
