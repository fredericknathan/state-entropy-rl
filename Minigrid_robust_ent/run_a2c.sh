#!/bin/bash
export PYTHONWARNINGS="ignore"
export WANDB_MODE=disabled
configs=(
  "beta=0.07 entropy_coef=0.01 ent_coef_decay=False"
  "beta=0.0 entropy_coef=0.01 ent_coef_decay=False"
  "beta=0.0 entropy_coef=0.0 ent_coef_decay=False"

)

max_parallel=4
count=0

for config in "${configs[@]}"; do
  eval $config
  for seed in {1..5}; do
    echo "Running: seed=$seed, beta=$beta, entropy=$entropy_coef"
    PYTHONPATH=./torch-ac python rl-starter-files/rl-starter-files/scripts/train.py \
      --env MiniGrid-FourRooms-v0 \
      --algo a2c \
      --seed $seed \
      --entropy-coef $entropy_coef \
      --ent_coef_decay $ent_coef_decay \
      --beta $beta \
      --procs 4 \
      --frames 3000000 &
    ((count++))
    if (( count % max_parallel == 0 )); then
      wait
    fi
  done
done

