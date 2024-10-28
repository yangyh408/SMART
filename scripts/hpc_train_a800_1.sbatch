#! /bin/bash


#SBATCH --job-name=SMART_8M_TRAINER

#SBATCH --partition=A800

#SBATCH --nodes=1

#SBATCH --gres=gpu:a800:1

#SBATCH --cpus-per-task=6

#SBATCH --mem=100G

#SBATCH --output=log/train_a800_1_%j.out

#SBATCH --error=log/train_a800_1_%j.err


module load cuda/11.3

source /share/apps/miniconda3/etc/profile.d/conda.sh

conda activate smart

cd ~/codes/SMART/

python train.py


