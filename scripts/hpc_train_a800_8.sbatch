#! /bin/bash


#SBATCH --job-name=SMART_8M_TRAINER

#SBATCH --partition=A800

#SBATCH --nodes=1

#SBATCH --gres=gpu:a800:8

#SBATCH --cpus-per-task=48

#SBATCH --mem=1000G

#SBATCH --time=48:00:00

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=yangyh408@tongji.edu.cn

#SBATCH --output=log/train_a800_8_%j.out

#SBATCH --error=log/train_a800_8_%j.err


module load cuda/11.3

source /share/apps/miniconda3/etc/profile.d/conda.sh

conda activate smart

cd ~/codes/SMART/

python train.py


