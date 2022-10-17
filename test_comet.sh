#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -p gypsum-2080ti  # Partition
#SBATCH -G 4  # Number of GPUs
#SBATCH --mem 80000

echo "SLURM_JOBID="$SLURM_JOBID
source "/home/yapeichang_umass_edu/rankgen/rankgen_venv/bin/activate"
echo "loaded venv"
python /home/yapeichang_umass_edu/rankgen/rankgen/rankgen_comet.py --num_shards 1