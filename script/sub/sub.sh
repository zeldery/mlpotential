#!/bin/bash
#
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#
#SBATCH --account=rrg-crowley-ac
#SBATCH --job-name=train
#SBATCH --output=train.log

source ~/working/bin/activate

# python pretrain.py
python train_charge.py -m runner_model.pt -e 5000 -d pbe0xdm-ani1x-hcno.h5 -g 1 -c checkpoint.pt -l 0.001


