#!/bin/sh
#SBATCH --job-name=job_<exp_id>_<local_rank>
#SBATCH -o /home/ella/rankgen/rankgen/parallel/parallel_logs/logs_exp_<exp_id>/log_<local_rank>.txt
#SBATCH --time=24:00:00
#SBATCH --partition=<gpu>
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH -d singleton

cd /home/ella/rankgen

<command> --local_rank <local_rank> --num_shards <total>
