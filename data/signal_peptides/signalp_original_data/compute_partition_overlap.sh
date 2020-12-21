
#USER=felteu

#!/bin/tcsh
#BSUB -n 1
#BSUB -J align
#BSUB -W 12:00
#BSUB -o  /work3/felteu/logs/check_identity/
#BSUB -e /work3/felteu/logs/check_identity/
#BSUB -q hpc
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=80GB]"

module load python3
source /zhome/1d/8/153438/experiments/py3-torch-env/bin/activate

python3 /zhome/1d/8/153438/experiments/master-thesis/data/signal_peptides/signalp_original_data/compute_partition_overlap.py