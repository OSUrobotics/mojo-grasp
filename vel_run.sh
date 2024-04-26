#!/bin/bash
#SBATCH -J PPOFullTimeTest                        # name of job
#SBATCH -p preempt                                  # name of partition or queue
#SBATCH --array=3-4                     # how many tasks in the array
#SBATCH -o log/LOG-%a.out                 # name of error file for this submission script
#SBATCH -e log/ERROR-%a.err                 # name of error file for this submission script
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 16
#SBATCH -t 1-00:00:00
# load any software environment module required for app (e.g. matlab, gcc, cuda)
#module load software/version
module load python3/3.8
source /nfs/hpc/share/swensoni/mojo-env/bin/activate
python3 --version
which python3 -a

# run my job (e.g. matlab, python)
srun --export=ALL /nfs/hpc/share/swensoni/mojo-env/bin/python3 run_hpc_multi.py $SLURM_ARRAY_TASK_ID 
