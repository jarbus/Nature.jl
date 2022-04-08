#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
 
# Load modules required for your job
module load cuda/9.0
module load anaconda
 
# Command to execute
