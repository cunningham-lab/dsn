#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=LRRNN
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --time=1:59:00
#SBATCH --mem-per-cpu=10gb

module load singularity

singularity exec --nv /moto/opt/singularity/tensorflow-1.13-gpu-py3-moto.simg python train_LRRNN.py $1 $2 $3 $4 $5
