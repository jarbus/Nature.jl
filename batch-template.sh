#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --time=02:00:00
#SBATCH --qos=low-gpu
#SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
 
# Load modules required for your job
module load cuda/9.0
module load anaconda
echo ". /opt/ohpc/pub/apps/installed/anaconda/5.2/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate julia
 
# Command to execute
