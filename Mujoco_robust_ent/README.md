
# Continuous-Control State-Entropy Regularization for Robust RL

This repository builds on the PPO implementation from **CleanRL** available at https://github.com/vwxyzjn/cleanrl.

---

## Requirements

This project is designed to run on **Linux** systems. To set up the environment:

```bash
conda env create -f env.yml
conda activate mujoco_robust
```

### OpenGL Requirements for Video Recording

If you plan to use `gym.record_video()` for capturing evaluation videos, you may need additional OpenGL packages:

```bash
# On Ubuntu/Debian
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3

# For headless systems (no display)
sudo apt-get install xvfb
```

**Important:** When running `video_capture`, you may need to set specific environment variables to resolve OpenGL-related issues. Please add the following to your environment before running video capture:

#### 1. Preload the system's `libstdc++.so.6`

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
```

This ensures the system Mesa library uses the newer system C++ standard library, enabling successful loading of `libOSMesa.so`.

#### 2. Force MuJoCo to use OSMesa backend

```bash
export MUJOCO_GL=osmesa
```

OSMesa (Off-Screen Mesa) is a pure software OpenGL implementation that can render without a display, making it ideal for WSL/headless server environments.

---

## Experiment Overview

This repository implements state-entropy regularization for continuous control tasks in MuJoCo environments. The experiments focus on two main tasks:

- **Pusher-v1**: A robotic arm manipulation task where the agent must push a puck to a target location
- **Ant-v1**: A quadruped robot locomotion task

The core idea is to maximize the entropy of specific state features (e.g., puck position in Pusher, body height/orientation in Ant) during training, encouraging the agent to explore diverse behaviors and improve robustness to environmental perturbations.

---

## Training

Use **`lstm_continuous_action_ppo.py`** to launch training. Key CLI flags:

| Flag | Description |
|------|-------------|
| `--ent_coef` | Policy-entropy coefficient (α) |
| `--starting_beta` | Initial state-entropy coefficient for warm-up |
| `--beta` | Final state-entropy coefficient (β) |
| `--beta_discount` | Discount factor for state-entropy intrinsic reward |
| `--ent_coef_decay` | Linearly anneal `ent_coef` to 0 during training (boolean) |
| `--num_envs` | Number of parallel environments; also sets batch size for state-entropy estimation |

---

## Experiment 1: Pusher

### Environment Overview

**Pusher** is a multi-joint robotic arm manipulation task. The robot controls a 7-DOF arm to push a cylindrical object (puck) to a designated target location using its fingertip.

**Environment**: `CustomPusher-v1` (modified from standard MuJoCo Pusher)

**State-entropy feature**: Puck (x, y) position (extracted from observation dimensions 17-18)

**Task characteristics**:
- Dense reward with shaped guidance (encourage reaching puck, then pushing to goal)
- Fixed initial state for reproducibility
- Episode length: 100 steps

### Action Space (7 continuous dimensions)

| Dimension | Joint | Control Range |
|-----------|-------|---------------|
| 0 | Shoulder translation | [-2, 2] N·m |
| 1 | Shoulder elevation | [-2, 2] N·m |
| 2 | Shoulder rotation | [-2, 2] N·m |
| 3 | Elbow flexion | [-2, 2] N·m |
| 4 | Forearm rotation | [-2, 2] N·m |
| 5 | Wrist flexion | [-2, 2] N·m |
| 6 | Wrist rotation | [-2, 2] N·m |

### Observation Space (23 dimensions)

| Indices | Description | Unit |
|---------|-------------|------|
| 0-6 | Arm joint positions (qpos) | radians |
| 7-13 | Arm joint velocities (qvel) | rad/s |
| 14-16 | End-effector position (x, y, z) | meters |
| 17-19 | Puck position (x, y, z) | meters |
| 20-22 | Target position (x, y, z) | meters |

### Reward Function

```
total_reward = reward_dist + reward_ctrl + reward_near
```

- **reward_near**: `-0.5 × ||fingertip_pos - puck_pos||₂` (encourage reaching)
- **reward_dist**: `-1.0 × ||puck_pos - goal_pos||₂` (main task reward)
- **reward_ctrl**: `-0.1 × ||action||₂²` (control cost penalty)

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 6 | Random seed |
| `--total_timesteps` | 5,000,000 | Total training timesteps |
| `--num_envs` | 32 | Parallel environments |
| `--num_steps` | 100 | Rollout length per update |
| `--learning_rate` | 6e-5 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae_lambda` | 0.95 | GAE lambda parameter |
| `--num_minibatches` | 4 | Mini-batch count |
| `--update_epochs` | 4 | PPO update epochs K |
| `--clip_coef` | 0.08 | PPO clipping coefficient ε |
| `--network_hidden_size` | 256 | Hidden layer size |
| `--beta_discount` | 0.91 | State-entropy discount factor |

### Experimental Conditions

Three training configurations are compared:

| Configuration | α (ent_coef) | β (beta) | Description |
|---------------|--------------|----------|-------------|
| No regularization | 0.0 | 0.0 | PPO baseline |
| Policy entropy only | 0.01 | 0.0 | Standard entropy regularization |
| State + Policy entropy | 0.01 | 160.0 (warm-up from 800.0) | Proposed method |

**State-entropy warm-up**: β follows cosine annealing from `starting_beta` to `beta` over training:

```
β_t = β + (β_start - β) / 2 × (1 + cos(π × (1 - t/T)))
```

### Training Script

To train the Pusher agent with the specified configurations, use the following commands:

#### No Regularization (Baseline)
```bash
python lstm_continuous_action_ppo.py --env CustomPusher-v1 --ent_coef 0.0 --beta 0.0 --total_timesteps 5000000 --num_envs 32
```

#### Policy Entropy Only
```bash
python lstm_continuous_action_ppo.py --env CustomPusher-v1 --ent_coef 0.01 --beta 0.0 --total_timesteps 5000000 --num_envs 32
```

#### State + Policy Entropy (Proposed Method)
```bash
python lstm_continuous_action_ppo.py --env CustomPusher-v1 --ent_coef 0.01 --starting_beta 800.0 --beta 160.0 --beta_discount 0.91 --total_timesteps 5000000 --num_envs 32
```

### Evaluation

Two evaluation scripts are provided to test agent robustness under different perturbation types:

#### 1. Wall Obstacle Evaluation (`Eval_pusher_wall.py`)

Tests robustness against an **L-shaped wall obstacle** (3 axis-aligned blocks) placed between the puck's initial position and the goal. The agent must navigate around the obstacle to complete the task.

**Usage:**
```bash
# Evaluate a trained model
python Eval_pusher_wall.py --model_path <path_to_model_dir> --output <output_csv_name>

# Example: Evaluate no regularization model
python Eval_pusher_wall.py --model_path runs_puck_final/CustomPusher-v1__2__alpha_0.0_beta_0.0_puck_True --output pusher_wall_no_reg.csv

# Example: Evaluate policy entropy only model
python Eval_pusher_wall.py --model_path runs_puck_final/CustomPusher-v1__2__alpha_0.01_beta_0.0_puck_True --output pusher_wall_policy_ent.csv

# Example: Evaluate state + policy entropy model
python Eval_pusher_wall.py --model_path runs_puck_final/CustomPusher-v1__2__alpha_0.01_beta_160.0_puck_True --output pusher_wall_state_ent.csv
```

**Evaluation details:**
- Tests both with and without wall obstacles
- Uses 15 test seeds by default
- 5 episodes per configuration
- Episode length: 100 steps
- Results saved to CSV with columns: `agent`, `seed`, `reward`, `distance`, `wall`, `success`

#### 2. Reward Shift Evaluation (`Eval_pusher_reward_shift.py`)

Tests robustness against **random goal position shifts**. The target location is shifted to a random position on a circle of radius r=0.15 around the original goal.

**Usage:**
```bash
# Evaluate a trained model
python Eval_pusher_reward_shift.py --model_path <path_to_model_dir> --output <output_csv_name>

# Example: Evaluate no regularization model
python Eval_pusher_reward_shift.py --model_path runs_puck_final/CustomPusher-v1__2__alpha_0.0_beta_0.0_grad_slow --output pusher_radius_no_reg.csv

# Example: Evaluate policy entropy only model
python Eval_pusher_reward_shift.py --model_path runs_puck_final/CustomPusher-v1__2__alpha_0.01_beta_0.0_grad_slow --output pusher_radius_policy_ent.csv

# Example: Evaluate state + policy entropy model
python Eval_pusher_reward_shift.py --model_path runs_puck_final/CustomPusher-v1__2__alpha_0.01_beta_160.0_grad_semi_slow --output pusher_radius_state_ent.csv
```

**Evaluation details:**
- Tests with goal shifted to random position on circle (radius=0.15)
- Uses 25 test seeds by default
- 100 episodes per configuration (more extensive testing)
- Episode length: 100 steps
- Results saved to CSV with columns: `agent`, `seed`, `reward`, `distance`, `radius`, `success`

#### Available Results

Pre-computed evaluation results are available as CSV files:

| File | Configuration | Evaluation Type |
|------|---------------|-----------------|
| `pusher_wall_no_reg.csv` | No regularization | Wall obstacle |
| `pusher_wall_policy_ent.csv` | Policy entropy only | Wall obstacle |
| `pusher_wall_state_ent.csv` | State + Policy entropy | Wall obstacle |
| `pusher_radius_no_reg.csv` | No regularization | Reward shift |
| `pusher_radius_policy_ent.csv` | Policy entropy only | Reward shift |
| `pusher_radius_state_ent.csv` | State + Policy entropy | Reward shift |

### Qualitative Results

The following videos demonstrate the behavior of each method on the Pusher task with wall obstacles:

#### No Regularization (α=0.0, β=0.0)

The robotic arm's end "bumped" at the white cylinde, and then immediately retracted to avoid it, failing to establish sustained contact to complete the push task.

<video src="videos_new/CustomPusher-v1__2__alpha_0.0__beta_0.0_puck_True/rl-video-episode-0.mp4" width="640" height="480" controls preload></video>

#### Policy Entropy Only (α=0.01, β=0.0)

The robotic arm's end continuously pressed the white cylinde against the red cube to form contact, resulting in a local stalemate.

<video src="videos_new/CustomPusher-v1__2__alpha_0.01__beta_0.0_puck_True/rl-video-episode-0.mp4" width="640" height="480" controls preload></video>

#### State Entropy + Policy Entropy (α=0.01, β=160.0)

The robotic arm bypassed the red obstacle and successfully pushed the white cylinder from the side to move it near the green target area.

<video src="videos_new/CustomPusher-v1__2__alpha_0.01__beta_160.0_puck_True/rl-video-episode-0.mp4" width="640" height="480" controls preload></video>

---

## Experiment 2: Ant

### Environment Overview

**Ant** is a quadruped robot locomotion task. The robot has a torso and four legs, each with two joints (hip and ankle). The goal is to coordinate leg movements to move forward along the positive x-axis.

**Environment**: `ANT-v1` (modified from MuJoCo Ant-v5)

**Key modifications** (to highlight regularization effects):
- Episode termination disabled when unhealthy (`terminate_when_unhealthy=False`)
- Removed healthy reward (`healthy_reward=0.0`)
- Disabled contact cost (`contact_cost_weight=0.0`)
- Disabled control cost (`ctrl_cost_weight=0.0`)
- Only forward progress reward remains

**State-entropy feature**: Body height (z-coordinate) and orientation (quaternion w, x, y, z)

**Rationale**: Maximizing height and orientation entropy encourages learning diverse locomotion gaits, enhancing adaptability.

### Action Space (8 continuous dimensions)

| Dimension | Joint | Control Range |
|-----------|-------|---------------|
| 0 | Right rear hip | [-1, 1] N·m |
| 1 | Right rear ankle | [-1, 1] N·m |
| 2 | Left front hip | [-1, 1] N·m |
| 3 | Left front ankle | [-1, 1] N·m |
| 4 | Right front hip | [-1, 1] N·m |
| 5 | Right front ankle | [-1, 1] N·m |
| 6 | Left rear hip | [-1, 1] N·m |
| 7 | Left rear ankle | [-1, 1] N·m |

### Observation Space (105 dimensions, excluding x, y position)

| Indices | Description | Unit |
|---------|-------------|------|
| 0 | Torso z-coordinate (height) | meters |
| 1-4 | Torso orientation (quaternion w, x, y, z) | - |
| 5-12 | 8 joint angles | radians |
| 13-15 | Torso velocity (vx, vy, vz) | m/s |
| 16-18 | Torso angular velocity | rad/s |
| 19-26 | 8 joint angular velocities | rad/s |
| 27-104 | External contact forces (13 positions × 6) | Newtons |

### Reward Function

**Modified** (shaped rewards removed for clarity):
```
reward = forward_reward
```

- **forward_reward**: `+1.0 × (x_after - x_before) / dt` (encourage forward motion)

**Original components removed**:
- `healthy_reward`: Default +1.0 for staying in healthy state
- `ctrl_cost`: `-0.5 × ||action||₂²`
- `contact_cost`: `-5e-4 × ||F_contact||₂²`

### Task Characteristics

- **Partial observability**: By default, x and y positions are excluded from observation
- **High-dimensional continuous control**: 8 action dimensions requiring coordination
- **State entropy benefit**: Diverse gaits learned through height/orientation entropy improve adaptability to obstacles and terrain changes

### Perturbation Setting

- **Vertical wall**: An invisible wall of size `[1, ∞, 0.3]` is placed in the environment; the ant must jump/hurdle over it
- **Evaluation protocol**: 200 episodes, each of length 100 steps

### Why Modify Rewards?

The standard Ant environment contains shaped rewards that already guide diverse behavior (e.g., healthy reward encourages stable locomotion). By removing these and keeping only the forward reward, we can better observe the effect of state entropy regularization, preventing regularization effects from being masked by shaped rewards.

