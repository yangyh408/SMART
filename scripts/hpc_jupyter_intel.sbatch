#! /bin/bash


#SBATCH --job-name=Jupyter

#SBATCH --partition=intel

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

#SBATCH --time=01:00:00

#SBATCH --output=log/jupyter_%j.out

export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}
port=10022

echo $node pinned to port $port
# print tunneling instructions jupyter-log

echo -e "
To connect to the compute node ${node} on sribd running your jupyter notebook server,
you need to run following two commands in a terminal
1. Command to create ssh tunnel from you workstation/laptop to cs-login:
ssh -N -f -L 127.0.0.1:${port}:${node}:${port} TJ_HPC
Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop
Use command behind to release bind
lsof -i :10022
kill -9 xxxx
"

# Run Jupyter

source /share/apps/miniconda3/etc/profile.d/conda.sh

conda activate smart

cd ~/codes/SMART/

jupyter lab --no-browser --port=${port}  --ip=${node} 

