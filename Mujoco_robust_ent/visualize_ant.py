# Ant environment visualization script with video capture
# Usage example:
#   python visualize_ant.py --model_path ant_run/ANT-v1__1__alpha_0.01_beta_160.0_starting_beta_800.0/agent.pth --eval_episodes 3
# Or:
#   python visualize_ant.py --load_dir ant_run --seed 1 --ent_coef 0.01 --beta 160.0 --starting_beta 800.0 --eval_episodes 3
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture_video", type=bool, default=True, help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--eval_steps", type=int, default=300, help="the number of steps to run in each environment per update")
    parser.add_argument("--env_id", default="ANT-v1", help="the id of the environment")
    parser.add_argument("--eval_episodes", type=int, default=3, help="the number of evaluation episodes")
    parser.add_argument("--num_envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--beta", type=float, default=160.0, help="coefficient of the state entropy")
    parser.add_argument("--starting_beta", type=float, default=800.0, help="starting beta value for locating the saved model folder")
    parser.add_argument("--network_hidden_size", type=int, default=256, help="the size of the hidden layer in the network")
    parser.add_argument("--functional_seed", type=int, default=101, help="the seed for the functional network")
    parser.add_argument("--load_dir", type=str, default="ant_all_reward", help="The trained agent's directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained agent .pth file (overrides load_dir)")
    parser.add_argument("--video_dir", type=str, default="videos_ant", help="the directory to save videos")
    parser.add_argument("--hurdles", type=bool, default=False, help="whether to use ant stairs (hurdles) environment")
    parser.add_argument("--healthy_reward", type=float, default=1.0, help="the reward for being healthy in the environment")

    args = parser.parse_args()
    args.wandb_group_name = f"action_entropy_{args.ent_coef}__state_entropy_{args.beta}"
    return args


def make_env(env_id, idx, capture_video, run_name, gamma, horizon=300, xml_file=None, healthy_reward=1.0, video_dir="videos_ant", exclude_current_positions_from_observation=False):
    def thunk():
        # Build kwargs dynamically, only include xml_file if it's not None
        base_kwargs = {
            "healthy_reward": healthy_reward,
            "exclude_current_positions_from_observation": exclude_current_positions_from_observation
        }
        if xml_file is not None:
            base_kwargs["xml_file"] = xml_file

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **base_kwargs)
            env = gym.wrappers.RecordVideo(env, f"{video_dir}/{run_name}", episode_trigger=lambda x: True)
        else:
            env = gym.make(env_id, **base_kwargs)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def normalize_obs(obs, mean, var):
    """Normalizes the observation using the running mean and variance of the observations."""
    epsilon = 1e-8
    return (obs - mean) / np.sqrt(var + epsilon)


if __name__ == "__main__":
    args = parse_args()
    args.repeats = args.eval_episodes // args.num_envs
    run_name = f"{args.env_id}__{args.seed}__alpha_{args.ent_coef}__beta_{args.beta}_hurdles_{args.hurdles}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.functional_seed)
    np.random.seed(args.functional_seed)
    torch.manual_seed(args.functional_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.hurdles:
        xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/ant_stairs.xml")
    else:
        xml_file = None  # use default ant.xml from gymnasium

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, horizon=args.eval_steps,
                  xml_file=xml_file, healthy_reward=args.healthy_reward, video_dir=args.video_dir,
                  exclude_current_positions_from_observation=False)
         for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)

    if args.model_path:
        model_path = args.model_path
    else:
        run_dir = f"{args.env_id}__{args.seed}__alpha_{args.ent_coef}_beta_{args.beta}_starting_beta_{args.starting_beta}"
        model_path = f"{args.load_dir}/{run_dir}/agent.pth"

    print(f"Loading model from: {model_path}")
    agent.load_state_dict(torch.load(model_path, weights_only=True))

    normalize_path = model_path.replace(".pth", "_normalize.npz")
    print(f"Loading normalization stats from: {normalize_path}")
    normalize = np.load(normalize_path, allow_pickle=True)
    norm_mean, norm_var = normalize["normalize_mean"].mean(), normalize["normalize_var"].mean()

    episode_returns = []
    for ep in range(args.eval_episodes):
        obs, _ = envs.reset(seed=args.functional_seed + ep)
        normed_obs = torch.Tensor(normalize_obs(obs, norm_mean, norm_var)).to(device)
        done = torch.zeros(args.num_envs).to(device)

        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        )

        ep_return = 0.0
        for step in range(args.eval_steps):
            with torch.no_grad():
                actions, logprob, _, value, next_lstm_state = agent.get_action_and_value(normed_obs, next_lstm_state, done)

            obs, reward, done, trunc, info = envs.step(actions.cpu().numpy())
            normed_obs = torch.Tensor(normalize_obs(obs, norm_mean, norm_var)).to(device)
            done = torch.Tensor(done).to(device)
            ep_return += reward.item()

        print(f"Episode {ep + 1}/{args.eval_episodes} return: {ep_return:.2f}")
        episode_returns.append(ep_return)

    envs.close()
    print(f"\nAverage return over {args.eval_episodes} episodes: {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
    print(f"Videos saved to: {args.video_dir}/{run_name}")
