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

    # Algorithm specific arguments
    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=2, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture_video", type=bool, default=True, help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--eval_steps", type=int, default=100, help="the number of steps to run in each environment per update")
    parser.add_argument("--env_id", default="CustomPusher-v1", help="the id of the environment")
    parser.add_argument("--eval_episodes", type=int, default=3, help="the number of evaluation episodes")
    parser.add_argument("--num_envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--beta", type=float, default=0.0, help="coefficient of the state entropy")
    parser.add_argument("--network_hidden_size", type=int, default=256, help="the size of the hidden layer in the network")
    parser.add_argument("--functional_seed", type=int, default=101, help="the seed for the functional network")
    parser.add_argument("--load_dir", type=str, default="runs_puck_final", help="The trained agent's directory")
    parser.add_argument("--wall_perturbation", type=bool, default=True, help="whether to use wall)

    args = parser.parse_args()
    args.wandb_group_name = f"action_entropy_{args.ent_coef}__state_entropy_{args.beta}"
    return args



def make_env(env_id, idx, capture_video, run_name, gamma, control_cost=0,horizon=100,xml_file=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id,reward_control_weight=control_cost, render_mode="rgb_array",xml_file=xml_file)
            env = gym.wrappers.RecordVideo(env, f"videos_new/{run_name}", episode_trigger=lambda x: True)
        else:
            env = gym.make(env_id,reward_control_weight=control_cost,xml_file=None)
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
    run_name = f"{args.env_id}__{args.seed}__alpha_{args.ent_coef}__beta_{args.beta}_puck_{args.wall_perturbation}"


    # TRY NOT TO MODIFY: seeding
    random.seed(args.functional_seed)
    np.random.seed(args.functional_seed)
    torch.manual_seed(args.functional_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.reapets = args.eval_episodes // args.num_envs
  
    if args.wall_perturbation:
        xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/custom_pusher_red_obstacle.xml")
    else: 
        xml_file=os.path.join(os.path.dirname(__file__), "mujoco_local/pusher_v5.xml")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma,horizon=args.eval_steps,xml_file=xml_file) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    path = f"{args.load_dir}/CustomPusher-v1__{args.seed}__alpha_{args.ent_coef}_beta_{args.beta}_grad_slow"
    agent.load_state_dict(torch.load(f"{path}/agent.pth", weights_only=True))
    normalize = np.load(f"{path}/agent_normalize.npz", allow_pickle=True)
    norm_mean, norm_var = normalize["normalize_mean"].mean(), normalize["normalize_var"].mean()

  
    episode_returns = []
    for ep in range(args.eval_episodes):
    #     envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name, args.gamma,horizon=args.eval_steps,xml_file=xml_file) for i in range(args.num_envs)]
    # )
        obs, _ = envs.reset(seed=args.functional_seed)
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

            
    envs.close()
                

        # episode_returns.append(ep_return)