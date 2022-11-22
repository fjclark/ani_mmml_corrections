#!/bin/bash
#SBATCH -o analysis-%A.%a.out
#SBATCH -p main
#SBATCH -n 1

srun /home/finlayclark/anaconda3/envs/mamba/envs/omm-beta-changed/bin/python repex_analysis.py --mmml_dir $1
