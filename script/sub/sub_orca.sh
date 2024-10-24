#!/bin/bash
#
#SBATCH --cpus-per-task=8
#
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#
#SBATCH --account=rrg-crowley-ac
#SBATCH --job-name=train
#SBATCH --output=train.log


source ~/working/bin/activate
module load StdEnv/2020 gcc/10.3.0 openmpi/4.1.1 orca/5.0.4
python datagen_compute.py -s scratch -i 0 -c control -t 1000
