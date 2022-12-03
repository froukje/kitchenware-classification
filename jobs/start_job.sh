#!/bin/bash
#SBATCH --job-name=ka
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=ka1176
#SBATCH --output=sr_out.o%j
#SBATCH --error=sr_err.e%j

conda init
source ~/.bashrc
conda activate kitchenware_classification
echo "conda env activated"

# Run script
codedir=/work/ka1176/frauke/kitchenware-classification
datadir=/work/ka1176/frauke/kitchenware-classification/data
logdir=/work/ka1176/frauke/kitchenware-classification/logs

#PYTHONPATH=$PYTHONPATH:"$codedir"
#export PYTHONPATH

python3 $codedir/train.py --gpus 1 --data $datadir  --logdir $logdir --num-worker 8 --max-epochs 100 --backbone resnet18
