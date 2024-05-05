#!/bin/bash -l
#SBATCH --chdir /home/mbricq/EE-559
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 01:30:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

# The --reservation line only works during the class.
conda activate EE559
echo $CONDA_PREFIX
echo "$PWD"
python src/run_fcm.py
