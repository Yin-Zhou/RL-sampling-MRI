#!/bin/sh
#
# RL submit script for Slurm.
#
#SBATCH --account=apam            # Replace ACCOUNT with your group account name bi

#SBATCH --job-name=RL             # The job name.
#SBATCH --gres=gpu:1              # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 32                     # The number of cpu cores to use
#SBATCH -t 0-12:00:00                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # The memory the job will use per cpu core

#SBATCH --mail-type=ALL           # select which email types will be sent
#SBATCH --mail-user=yz3888@columbia.edu # NOTE: put your netid here if you want emails
#SBATCH --output=r_%A_%a.out      # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=r_%A_%a.err
 
module load anaconda
conda init bash
source ~/.bashrc
conda activate rl

touch RUNNING

echo ${SLURM_ARRAY_TASK_ID}
echo "Launching a Python run"
date
 
#Command to execute Python program
python ppo_SP.py

touch done

\rm RUNNING
 
#End of script