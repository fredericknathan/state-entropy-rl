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
    parser.add_argument("--load_dir", type=str, default="runs_puck_final", help="The trained agent's directory")
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

def make_env(env_id, idx, run_name, gamma,horizon=300,xml_file=None,healthy_reward=0.0):
    def thunk():
        env = gym.make(env_id,xml_file=xml_file,healthy_reward=healthy_reward)
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

    configs = [{"beta": 200.0, "ent_coef": 0.001,"starting_beta": 0.0},
               {"beta": 0.0, "ent_coef": 0.0,"starting_beta": 0.0},
               {"beta": 0.0, "ent_coef": 0.001,"starting_beta": 0.0}]
   
    
    df = pd.DataFrame(columns=['agent','seed' ,'reward','distance','hurdels_passed', 'hurdels'])

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.reapets = args.eval_episodes // args.num_envs
  
    for hurdels in [True,False]:      
        if hurdels:
            xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/ant_stairs.xml")
        else:
            xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/ant.xml")

        envs = gym.vector.SyncVectorEnv(
                        [make_env(args.env_id, i, run_name, args.gamma, xml_file=xml_file,horizon=args.eval_steps,healthy_reward=args.healthy_reward) for i in range(args.num_envs)]
                    )
        for seed in args.test_seeds:#2,
            for i, conf in tqdm(enumerate(configs)):
                agent = Agent(envs).to(device)
                path = f"ant_all_reward/ant_beta_{conf['beta']}_alpha_{conf['ent_coef']}_starting_beta_{conf['starting_beta']}/ANT-v1__{seed}__alpha_{conf['ent_coef']}_beta_{conf['beta']}_starting_beta_{conf['starting_beta']}"
                agent.load_state_dict(torch.load(f"{path}/agent.pth", weights_only=True))
                normalize = np.load(f"{path}/agent_normalize.npz", allow_pickle=True)
                norm_mean, norm_var = normalize["normalize_mean"].mean(), normalize["normalize_var"].mean()

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

                    episode_rewards.append(cumulative_rewards.mean().item())
                    episode_hurdles.append(final_hurdles.mean().item())
                    episode_distances.append(final_distances.mean().item())
                    episode_successes.append(successes.mean().item())


                avg_hurdles = np.mean(episode_hurdles)
                avg_reward = np.mean(episode_rewards)
                avg_distance = np.mean(episode_distances)
                avg_success = np.mean(episode_successes)
                df.loc[len(df)] = [f"beta:{conf['beta']}_alpha:{conf['ent_coef']}_starting_beta{conf['starting_beta']}", seed, avg_reward, avg_distance, avg_hurdles, hurdels]


    df.to_csv('ant_all_rewards_hurdels_hard_25.csv', index=False)
   
