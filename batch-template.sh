#!/bin/bash
#SBATCH --account=guest
#SBATCH --partition=guest-gpu
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:TitanX:1   # Request 1 TitanX GPU
#SBATCH -q all.q      ### specify which queue you want
#SBATCH -ckpt reloc   ### the job will be relocated upon suspension.
#SBATCH -cwd          ### tells the system you want to start  from the same folder
                 ### you were when you submitted the job

source /home/garbus/.bashrc
conda activate julia
