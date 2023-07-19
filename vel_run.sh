#!/bin/bash
#SBATCH -J PPOInHandManipulation                        # name of job
#SBATCH -p dgx2                                  # name of partition or queue
#SBATCH --array=1-3                     # how many tasks in the array
#SBATCH -o log/LOG-%a.out                 # name of error file for this submission script
#SBATCH -e log/ERROR-%a.err                 # name of error file for this submission script
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --constraint=skylake
# load any software environment module required for app (e.g. matlab, gcc, cuda)
#module load software/version
#module load python3
source /nfs/hpc/share/swensoni/mojo-env/bin/activate
# run my job (e.g. matlab, python)
srun --export=ALL python3 run_hpc_PPO.py $SLURM_ARRAY_TASK_ID 
