#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --mincpus=20
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G

#SBATCH --time=4-00:00:00
#SBATCH --output=cumulative.txt
#SBATCH --mail-user=ansh.puvvada@students.iiit.ac.in
#SBATCH --mail-type=ALL


eval "$(conda shell.bash hook)"
conda activate venv

# /home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_trainer c_n2_r1_m6

# /home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_tester c_n2_r1_m6

# /home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_tester c_n3_r2_m4

# /home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_trainer c_n3_r1_m4

/home/lalitha.v/miniconda3/envs/venv/bin/python3.8 -m app.auto_tester c_n3_r1_m4
