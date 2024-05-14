#!/bin/bash
#SBATCH -J PPORotate                        # name of job
#SBATCH -p dgx2                                  # name of partition or queue
#SBATCH --array=1-6                     # how many tasks in the array
#SBATCH -o log/LOG-%a.out                 # name of error file for this submission script
#SBATCH -e log/ERROR-%a.err                 # name of error file for this submission script
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 16
#SBATCH -t 2-00:00:00

# load any software environment module required for app (e.g. matlab, gcc, cuda)
#module load software/version
source /nfs/hpc/share/swensoni/virtual_mojo/bin/activate

# run my job (e.g. matlab, python)
srun --export=ALL python3 run_hpc_multi.py $SLURM_ARRAY_TASK_ID 
