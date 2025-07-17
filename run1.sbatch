#!/bin/bash
#SBATCH --job-name=4_MBDG
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#NOSBATCH --partition=hpg-b200
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chenz1@ufl.edu

cd /home/chenz1/toorange/FDDG

bash datasetsGo.sh
