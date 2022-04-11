#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=run.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --time=24:00:00
#SBATCH --qos=low-gpu
#SBATCH --gres=gpu:V100:1
 
# Load modules required for your job
conda init bash
source /home/garbus/.bashrc
conda activate julia
 
# Command to execute
