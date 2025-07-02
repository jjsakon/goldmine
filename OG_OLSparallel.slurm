#!/bin/bash
#SBATCH --output=/home1/john/Goldmine/time_cells_goldmine/cluster/log-%j.out
#SBATCH --error=/home1/john/Goldmine/time_cells_goldmine/cluster/log-%j.err
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --array=4-451%50 # for 450 neurons

echo "Running task number $SLURM_ARRAY_TASK_ID"
python -u /home1/john/Goldmine/time_cells_goldmine/OLSparallel.py $SLURM_ARRAY_TASK_ID