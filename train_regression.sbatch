#!/bin/bash

#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=dgx
#SBATCH --cpus-per-gpu=4


source activate object_tracking

#python test_regression.py --method="DKT" --output_dim=2 --seed=1 --dataset sines --save_dir ./save/regression_sines_nn --kernel_type nn --stop_epoch 10000
python test_regression.py --method="DKT" --output_dim=2 --seed=1 --dataset sines --save_dir ./save/regression_sines_nn --kernel_type nn
