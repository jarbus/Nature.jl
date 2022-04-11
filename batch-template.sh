#!/bin/bash
#SBATCH --job-name=OEC
#SBATCH --output=run.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
 
# Load modules required for your job
source /home/garbus/.bashrc
conda activate julia
 
# Command to execute
