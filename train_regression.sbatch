#!/bin/bash

#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=dgx
#SBATCH --cpus-per-gpu=4


source activate object_tracking

python train_regression.py --method="DKT" --output_dim=1 --seed=1 --dataset sines
#python test_regression.py --method="DKT" --output_dim=1 --seed=1 --dataset sines