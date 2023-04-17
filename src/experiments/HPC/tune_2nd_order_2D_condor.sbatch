#!/bin/bash

# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long

# The default run (wall-clock) time is 1 minute
#SBATCH --time=24:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=4

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=10000

# Add GPU
#SBATCH --gres=gpu

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# Use this simple command to check that your sbatch settings are working
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

# Uncomment these lines when your job requires this software
module use /opt/insy/modulefiles
module load cuda/11.2 cudnn/11.2-8.1.1.33
export PYTHONPATH=$PYTHONPATH:/home/nfs/rperezdattari/pcgrad

srun python3 run_optuna.py --params 2nd_order_2D_condor --hyperparameter-optimization True --results-base-directory /tudelft.net/staff-umbrella/condorstorage/riemannian/

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"