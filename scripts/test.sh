#!/bin/bash -l
#SBATCH --chdir /scratch/izar/koehn
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 2:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559
#SBATCH --reservation ee-559
# The --reservation line only works during the class.
conda activate dl
python src/eval.py
