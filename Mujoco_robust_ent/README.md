

# Continuous-Control State-Entropy Regularization for Robust RL

This repository builds on the PPO implementation from **CleanRL** available at  https://github.com/vwxyzjn/cleanrl.

---

## Requirements

```bash
conda env create -f env.yml
```

## Training

Use **`lstm_continuous_action_ppo.py`** to launch training. Key CLI flags:

- **`--ent_coef`** – policy-entropy coefficient.
- **`--starting_beta`** – initial state-entropy coefficient.
- **`--beta`** – final state-entropy coefficient (where `starting_beta` decays to).
- **`--beta_discount`** – discount factor for the state-entropy intrinsic reward.
- **`--ent_coef_decay`** – linearly anneal `ent_coef` to 0 over the course of training (boolean switch).
- **`--num_envs`** – number of parallel environments; also sets the number of rollouts per PPO update and the batch size used for state-entropy estimation.


## Evaluation

The “wall” experiment uses **'Eval_pusher_wall.py'**. To train and evaluate:

```eval
bash train_and_evaluate.sh
```
hyperparameters can be specified in the configs variable inside.
the evaluation results will be saved in a .csv file

 
