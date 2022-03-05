#!/bin/bash

# Slurm sbatch options
#SBATCH -o logs/benchmark.log-%j
#SBATCH -c 10
# #SBATCH --gres=gpu:volta:1

# Initialize the module first
source /etc/profile

# Load Anaconda
module load anaconda/2022a

# Activate environment
source activate orion-2.0-env

# Python executable
python experiments.py
