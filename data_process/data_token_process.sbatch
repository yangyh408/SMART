#! /bin/bash


#SBATCH --job-name=Data_Process

#SBATCH --partition=intel

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

#SBATCH --array=0-47

#SBATCH --output=data_process_%A_%a.out

source /share/apps/miniconda3/etc/profile.d/conda.sh

conda activate smart

cd ~/codes/SMART/

python data_token_process.py

