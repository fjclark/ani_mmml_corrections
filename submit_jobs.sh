#!/bin/bash
#SBATCH -o ani_correction-%A.%a.out
#SBATCH -p main
#SBATCH -n 1
#SBATCH --gres=gpu:1

echo "CUDA DEVICES:" $CUDA_VISIBLE_DEVICES
echo "LIG_NAME is: " $1

srun /home/finlayclark/anaconda3/envs/mamba/envs/omm-beta-changed/bin/python ../../../run_cor.py --lig_name $1 --n_iter $2 --n_states $3 --pdb_path $4
