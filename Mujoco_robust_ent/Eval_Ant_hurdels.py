# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from normalize import NormalizeObservation
import utils
import argparse
from lstm_continuous_action_ppo import Agent
from tqdm import tqdm
import pandas as pd
import mujoco_local
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm specific argumentsObstaclePusher-v0
    #fill the arguments of the policy you wish to evaluate
    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=10, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--eval_steps", type=int, default=300, help="the number of steps to run in each environment per update")
    parser.add_argument("--env_id", default="ANT-v1", help="the id of the environment")
    parser.add_argument("--eval_episodes", type=int, default=100, help="the number of evaluation episodes")
    parser.add_argument("--num_envs", type=int, default=20, help="the number of parallel game environments")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--beta", type=float, default=160.0, help="coefficient of the state entropy")
    parser.add_argument("--starting_beta", type=float, default=800.0, help="coefficient of the state entropy warmup")
    parser.add_argument("--test_seeds", type=list, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], help="the evaluation seeds")
    parser.add_argument("--network_hidden_size", type=int, default=256, help="the size of the hidden layer in the network")
    parser.add_argument("--healthy_reward", type=float, default=0.0, help="the reward for being healthy in the environment")
    parser.add_argument("--model_path", type=str, required=True, help="The trained agent's model directory path")
    parser.add_argument("--output", type=str, default="ant_hurdels_results.csv", help="Output CSV file name")
    # to be filled in runtime

    args = parser.parse_args()
    args.num_envs = min(args.num_envs, args.eval_episodes)  # make sure num_envs is not larger than eval_episodes
    args.wandb_group_name = f"action_entropy_{args.ent_coef}__state_entropy_{args.beta}_fixed_reset"
    return args

# hurdle_edges = [4.5, 10.5, 16.5, 22.5]
hurdle_edges = [18.5]
def count_hurdles(x_pos):
    # count how many edges the final x crosses
    return sum([1 for edge in hurdle_edges if x_pos > edge])

def make_env(env_id, idx, run_name, gamma,horizon=300,xml_file=None,healthy_reward=0.0,exclude_current_positions_from_observation=False):
    def thunk():
        # Build kwargs dynamically, only include xml_file if it's not None
        kwargs = {
            "healthy_reward": healthy_reward,
            "exclude_current_positions_from_observation": exclude_current_positions_from_observation
        }
        if xml_file is not None:
            kwargs["xml_file"] = xml_file
        env = gym.make(env_id, **kwargs)
        env=gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))


        return env

    return thunk

def normalize_obs(obs, mean, var):
    """Normalizes the observation using the running mean and variance of the observations."""
    epsilon = 1e-8  # small constant to avoid division by zero
    return (obs - mean) / np.sqrt(var + epsilon)

if __name__ == "__main__":
    args = parse_args()
    args.repeats = args.eval_episodes // args.num_envs
    run_name = f"{args.env_id}__{args.seed}__alpha_{args.ent_coef}__beta_{args.beta}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    df = pd.DataFrame(columns=['agent','seed' ,'reward','distance','hurdels_passed', 'hurdels'])

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.reapets = args.eval_episodes // args.num_envs

    # Calculate total iterations for progress bar
    total_iterations = len([True, False]) * len(args.test_seeds) * args.repeats
    total_steps = total_iterations * args.eval_steps

    print(f"Starting evaluation:")
    print(f"  - Seeds: {len(args.test_seeds)}")
    print(f"  - Episodes per seed: {args.repeats}")
    print(f"  - Steps per episode: {args.eval_steps}")
    print(f"  - Total episodes: {total_iterations}")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Parallel environments: {args.num_envs}")
    print()

    with tqdm(total=total_iterations, desc="Evaluating") as pbar:
        for hurdle_idx, hurdles in enumerate([True, False]):
            hurdle_desc = "with hurdles" if hurdles else "without hurdles"
            pbar.set_description(f"Testing {hurdle_desc}")

            if hurdles:
                xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/ant_stairs.xml")
            else:
                xml_file = None  # Use Gymnasium's default ant.xml

            envs = gym.vector.SyncVectorEnv(
                            [make_env(args.env_id, i, run_name, args.gamma, xml_file=xml_file,horizon=args.eval_steps,healthy_reward=args.healthy_reward,exclude_current_positions_from_observation=False) for i in range(args.num_envs)]
                        )

            # Load model once per hurdle configuration (not per seed)
            agent = Agent(envs).to(device)
            agent.load_state_dict(torch.load(f"{args.model_path}/agent.pth", weights_only=True))
            normalize = np.load(f"{args.model_path}/agent_normalize.npz", allow_pickle=True)
            norm_mean, norm_var = normalize["normalize_mean"].mean(), normalize["normalize_var"].mean()

            for seed_idx, seed in enumerate(args.test_seeds):
                episode_rewards = []
                episode_distances = []
                episode_successes = []
                episode_hurdles = []
                for k in range(args.repeats):
                    obss, _ = envs.reset(seed=args.seed+k)
                    obss = torch.Tensor(normalize_obs(obss, norm_mean, norm_var)).to(device)
                    next_lstm_state = (
                        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                    )
                    dones = torch.zeros(args.num_envs, device=device)
                    step = 0
                    cumulative_rewards = torch.zeros(args.num_envs, device=device)
                    final_distances = torch.zeros(args.num_envs, device=device)
                    final_hurdles = torch.zeros(args.num_envs, device=device)
                    successes = torch.zeros(args.num_envs, device=device)
                    mask = torch.ones(args.num_envs, device=device)
                    while step < args.eval_steps:
                        with torch.no_grad():
                            actions, logprob, _, value, next_lstm_state = agent.get_action_and_value(obss, next_lstm_state, dones)

                        obss, rewards, dones, truncations, infos = envs.step(actions.cpu().numpy())
                        obss = torch.Tensor(normalize_obs(obss, norm_mean, norm_var)).to(device)
                        dones = torch.Tensor(dones).to(device)
                        rewards = torch.Tensor(rewards).to(device)

                        cumulative_rewards += rewards * mask

                        for idx, done in enumerate(dones):
                            if done or truncations[idx]:
                                mask[idx] = 0
                                if 'final_info' in infos:
                                    final_distances[idx] = infos['final_info'][idx]['x_position']
                                else:
                                    final_distances[idx] = infos['x_position'][idx]

                                final_hurdles[idx] = count_hurdles(final_distances[idx])


                        step += 1

                    # End of one episode - record results and update progress
                    episode_rewards.append(cumulative_rewards.mean().item())
                    episode_hurdles.append(final_hurdles.mean().item())
                    episode_distances.append(final_distances.mean().item())
                    episode_successes.append(successes.mean().item())

                    # Update progress bar
                    pbar.update(1)
                    episodes_completed = (hurdle_idx * len(args.test_seeds) * args.repeats +
                                         seed_idx * args.repeats + (k + 1))
                    steps_completed = episodes_completed * args.eval_steps
                    pbar.set_postfix({'seed': seed, 'hurdles': hurdle_desc,
                                     'episodes': f'{episodes_completed}/{total_iterations}',
                                     'steps': f'{steps_completed:,}/{total_steps:,}'})

                # End of all episodes for this seed - compute averages and save to CSV
                avg_hurdles = np.mean(episode_hurdles)
                avg_reward = np.mean(episode_rewards)
                avg_distance = np.mean(episode_distances)
                avg_success = np.mean(episode_successes)
                model_name = os.path.basename(args.model_path)
                df.loc[len(df)] = [model_name, seed, avg_reward, avg_distance, avg_hurdles, hurdles]

    df.to_csv(args.output, index=False)

    # Calculate and print average results
    avg_hurdles_with = df[df['hurdels'] == True]['hurdels_passed'].mean()
    avg_hurdles_without = df[df['hurdels'] == False]['hurdels_passed'].mean()
    print(f"Model: {args.model_path}")
    print(f"Average hurdles passed (with hurdles): {avg_hurdles_with:.4f}")
    print(f"Average hurdles passed (without hurdles): {avg_hurdles_without:.4f}")
   
