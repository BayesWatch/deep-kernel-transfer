#!/bin/bash

#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=dgx
#SBATCH --cpus-per-gpu=4


source activate object_tracking

python run_regression.py \
--test \
--neptune \
--dataset $2 \
--model=$3 \
--method="DKT" \
--output_dim=1 \
--seed=1 \
--save_dir ./save/"regression_$2_$1" \
--kernel_type $1 \
--stop_epoch 1000
