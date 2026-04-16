#!/bin/bash
export PYTHONWARNINGS="ignore"
export WANDB_MODE=disabled

rm -rf rl-starter-files/rl-starter-files/scripts/utils/STORAGE_NEW/

echo "Running quick test: 1 seed, 5000 frames"
PYTHONPATH=./torch-ac python rl-starter-files/rl-starter-files/scripts/train.py \
  --algo a2c \
  --seed 1 \
  --entropy-coef 0.01 \
  --ent_coef_decay False \
  --beta 0.07 \
  --procs 1 \
  --frames 5000

echo "Done!"
