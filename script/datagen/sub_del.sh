#!/bin/bash
#
#SBATCH --cpus-per-task=3
#
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#
#SBATCH --account=rrg-crowley-ac

rm -r scratch
mkdir scratch

