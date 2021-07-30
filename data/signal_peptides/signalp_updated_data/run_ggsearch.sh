#USER=felteu
experiment_name=align_signalp6



#!/bin/tcsh
#BSUB -n 1
#BSUB -J ggsearch
#BSUB -W 72:00
#BSUB -o  /work3/felteu/logs/tmp/
#BSUB -e /work3/felteu/logs/tmp/
#BSUB -q hpc
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=100GB]"

module load python3
python3 get_edgelist.py
