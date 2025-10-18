import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import numpy as np
import utils
from utils.custom_env import EmptyEnv
from tqdm import tqdm

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='MiniGrid-EmptyEnv-v0',
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model",default='MiniGrid-EmptyEnv-v0_seed1_True_False_beta0.05_entropy0.02',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    env = utils.make_env(args.env, args.seed + 10000 * i)
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agentVCSE_A2C/rl-starter-files/rl-starter-files/scripts/STORAGE_NEW/
for i in tqdm(range(1,6)):
    args.model = 'MiniGrid-EmptyEnv-v0_seed'+str(i)+'_True_False_beta0.05_entropy0.02'
    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        device=device, argmax=args.argmax, num_envs=args.procs,
                        use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)
    logs_risks_taken = torch.zeros(args.procs, device=device)
    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, info = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        # logs_risks_taken += torch.tensor([info[i]['riskes_taken'] for i in range(args.procs)], device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)


        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames/(end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    # print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
    #     .format(num_frames, fps, duration,
    #             *return_per_episode.values(),
    #             *num_frames_per_episode.values()))

    # Print worst episodes

    n = args.episodes 
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
        
    #calculate the avg length of rhe episodes with a positive reward:
    positive_rewards = [logs["num_frames_per_episode"][i] for i in range(len(logs["return_per_episode"])) if logs["return_per_episode"][i] > 0]
    avg_positive_reward = np.mean(positive_rewards)
    print("The average length of the episodes with a positive reward is: ", avg_positive_reward)
    print("success rate: ", len(positive_rewards)/args.episodes)    
    print("The average number of risks taken is: ", torch.mean(logs_risks_taken).item())
    #get the success rate at each time step:
    success_rate = []
    #longest episode:
    longest_episode = int(max(logs["num_frames_per_episode"]))
    for i in range(1, longest_episode+1):
        succseeded_in_i_steps = sum([1 for j in range(n) if logs["num_frames_per_episode"][j] <= i])
        success_rate.append(succseeded_in_i_steps/n)

    save_path = f'/home/yonatanashlag/RobustEnt/VCSE/VCSE_A2C/rl-starter-files/rl-starter-files/scripts/Evaluations2/NOMINAL{args.model}'
    #make success rate a numpy array:
    success_rate = np.array(success_rate)
    np.save(save_path, success_rate)
