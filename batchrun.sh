#!/bin/bash
#SBATCH -J leakyTest2                       # name of job
#SBATCH -p dgxs                                  # name of partition or queue
#SBATCH -o log/LOG-%a.out                 # name of error file for this submission script
#SBATCH -e log/ERROR-%a.err                 # name of error file for this submission script
#SBATCH --time=11:00:00
#SBATCH -c 4
#SBatch --mem=8G
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
# load any software environment module required for app (e.g. matlab, gcc, cuda)

#module load software/version
#module load python3

source /nfs/hpc/share/navek/nigel_rl/mojo-grasp/venv/bin/activate
# run my job (e.g. matlab, python)

srun --export=ALL python3 run_hpc.py 0
