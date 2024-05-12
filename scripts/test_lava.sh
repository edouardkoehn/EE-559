#!/bin/bash -l
#SBATCH --chdir /home/koehn/EE-559
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --time 04:00:00
#SBATCH --gres gpu:1
#SBATCH --account ee-559
#SBATCH --qos ee-559

# Initialize the env
conda activate EE559
echo $CONDA_PREFIX
echo "$PWD"

#Set up the command
config_file="config_Lava0S_tweet_only.json"
command="run_lava -c $config_file"
echo "$comand"

#run the code
$command
