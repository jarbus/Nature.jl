#!/bin/bash
#SBATCH --job-name=OEC
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU

source /home/garbus/.bashrc
conda activate julia
