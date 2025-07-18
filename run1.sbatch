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


PYTHONPATH=/home/chenz1/toorange/FDDG/fddg uv run python -m train_munit.py  --env 0 --device 0 --dataset FairFace --step 12 --output_path /home/chenz1/toorange/FDDG/fddg/domainbed/munit/FairFace --input_path /home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/FairFace/pretrain_env0_step1/outputs/tiny_munit/checkpoints --input_path1 /home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/FairFace/pretrain_env0_step1/outputs/tiny_munit/checkpoints --input_path2 /home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/FairFace/pretrain_env0_step2/outputs/tiny_munit/checkpoints --resume

PYTHONPATH=/home/chenz1/toorange/FDDG/fddg uv run python -m train_munit.py  --env 0 --device 0 --dataset YFCC --step 12 --output_path /home/chenz1/toorange/FDDG/fddg/domainbed/munit/YFCC --input_path /home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/YFCC/pretrain_env0_step1/outputs/tiny_munit/checkpoints --input_path1 /home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/YFCC/pretrain_env0_step1/outputs/tiny_munit/checkpoints --input_path2 /home/chenz1/toorange/FDDG/fddg/domainbed/munit/munit_results_new/models/YFCC/pretrain_env0_step2/outputs/tiny_munit/checkpoints
