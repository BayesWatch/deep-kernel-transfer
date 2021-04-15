#!/bin/bash

#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=dgx
#SBATCH --cpus-per-gpu=4


source activate object_tracking

python run_regression.py \
--flow \
--neptune \
--model=$3 \
--method="DKT" \
--output_dim=1 \
--seed=1 \
--dataset $2 \
--save_dir ./save/"regression_flow_$2_$1" \
--kernel_type $1 \
--stop_epoch 1000 \
--use_conditional False
