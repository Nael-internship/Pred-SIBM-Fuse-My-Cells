#!/bin/bash

#SBATCH -p XXXXX -A YYYYYYYYY
#SBATCH -N 1                       # 1 node
#SBATCH -n 1                       # 1 task
#SBATCH --mem=16G                  # 16Go RAM

hostname
nvidia-smi

python --version

METHOD=$1
ARGS=$2

echo "Running case: method=$METHOD, args=$ARGS"
python eval.py --method $METHOD --args $ARGS --dataset /home/uha/cmeyer/data/FuseMyCells.hdf5
echo "Finished run"

echo '-- THIS IS THE END --'
