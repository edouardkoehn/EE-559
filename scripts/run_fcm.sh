#!/bin/bash -l
#SBATCH --chdir /home/mbricq/EE-559
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time 03:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

# The --reservation line only works during the class.
conda activate EE559
echo $CONDA_PREFIX
echo "$PWD"
python src/run_fcm.py
