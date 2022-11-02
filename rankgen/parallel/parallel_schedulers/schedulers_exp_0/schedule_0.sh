#!/bin/sh
#SBATCH --job-name=job_0_0
#SBATCH -o /home/ella/rankgen/rankgen/parallel/parallel_logs/logs_exp_0/log_0.txt
#SBATCH --time=24:00:00
#SBATCH --partition=2080ti-short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH -d singleton

cd /home/ella/rankgen

python rankgen/comet_beam_search.py --num_tokens 8 --local_rank 0 --num_shards 8

