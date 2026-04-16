import sys
import os
import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX

import csv
import wandb
import gym
import utils
from model import ACModel
from utils.custom_env import EmptyEnv

import numpy as np
from torch_ac.utils.penv import ParallelEnv
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    #MiniGrid-CrossRoads-v0
    #MiniGrid-SimpleCrossingS9N1-v0
    ## General parameters
    parser.add_argument("--algo",default='a2c',
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", default='MiniGrid-EmptyEnv-v0',
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=57,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=8,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=3*10**6,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=100,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--ent_coef_decay",type=bool , default=True,
                        help="decay the ent_coef through training (default: False)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")


    parser.add_argument("--use_entropy_reward", action="store_true", default=True)
    parser.add_argument("--use_value_condition", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--use_batch", action="store_true", default=True)

    args = parser.parse_args()
    args.wandb_group = f"beta:{args.beta}_alpha:{args.entropy_coef}"
    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_seed{args.seed}_beta_{args.beta}alpha_{args.entropy_coef}"
    #wandb:
    wandb.init(project='RobustEnt_Minigrid_new', name=default_model_name,id=default_model_name,resume="allow")
    #make sure that the configs is uploaded to wandb:
    wandb.config.update(vars(args))
    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    #sync wanbd with tensorboard
    #wandb.tensorboard.patch(tb_writer)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss, 
                                use_entropy_reward=args.use_entropy_reward,use_value_condition=args.use_value_condition)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    if algo.use_true_pos:
        algo.replay_buffer = np.zeros((10000, 6))
    else:
        algo.replay_buffer = np.zeros((10000, 64))
    algo.idx = 0
    algo.full = False
    ent_coef = args.entropy_coef
    algo.beta = args.beta
    algo.use_batch = args.use_batch
    args.num_updates = int(args.frames / args.frames_per_proc / args.procs)
    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        if args.use_entropy_reward:
            for i in range(len(exps.obs)):
                # store low-dimensional features from random encoder
                if algo.use_true_pos:
                    algo.replay_buffer[algo.idx] = exps.agent_pos[i].detach().cpu().numpy()
                else:
                    algo.replay_buffer[algo.idx] = algo.random_encoder(exps.obs[i].image.unsqueeze(0).transpose(1, 3).transpose(2, 3))[0,:,0,0].detach().cpu().numpy()
                algo.idx = (algo.idx + 1) % algo.replay_buffer.shape[0]
                algo.full = algo.full or algo.idx == 0

        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1
        #lenearily interpolate to zero the self.entropy_coef:
        if args.ent_coef_decay:
            frac = 1.0 - (update - 1.0) / args.num_updates
            ent_now = frac * args.entropy_coef
            algo.entropy_coef = ent_now



        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:usmM {:.2f} {:.2f} {:.2f} {:.2f} | F:usmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | g {:.3f}"
                .format(*data))
            

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
            #log the data to wandb:
            wandb.log({header[i]:data[i] for i in range(len(header))})
            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if (args.save_interval > 0 and update % args.save_interval == 0):
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

    # Evaluate model kernel robustness
    #create test environments:

    eval_procs = 1
    agent = utils.Agent(envs[0].observation_space, envs[0].action_space, model_dir,
                    device=device, argmax=False, num_envs=eval_procs,
                    use_memory=False, use_text=False)
    agent.acmodel=algo.acmodel
    args.porcs =20
    args.episodes = 100
    eval_tasks = {1:"wall",2:"elbow",3:"random_blocks",4:"reward_shift"}
    for task in tqdm(range(4)): #different blocked modes
        test_envs = []  
        for j in range(eval_procs):
            env = utils.make_env(args.env,1000 *task+j, blocked=task+1)
            test_envs.append(env)
        test_env = ParallelEnv(test_envs)

        logs = {"num_frames_per_episode": [], "return_per_episode": []}

        # Run agent
        log_episode_return = torch.zeros(eval_procs, device=device)
        log_episode_num_frames = torch.zeros(eval_procs, device=device)
        obss = test_env.reset()
        log_done_counter = 0
        
        while log_done_counter < args.episodes:
            actions = agent.get_actions(obss)
            obss, rewards, dones, info = test_env.step(actions)
            agent.analyze_feedbacks(rewards, dones)

            log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
            log_episode_num_frames += torch.ones(eval_procs, device=device)

            for i, done in enumerate(dones):
                if done:
                    log_done_counter += 1
                    logs["return_per_episode"].append(log_episode_return[i].item())
                    logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

            mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
            log_episode_return *= mask
            log_episode_num_frames *= mask



        # Print logs

        num_frames = sum(logs["num_frames_per_episode"])
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        # print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
        #     .format(num_frames, fps, duration,
        #             *return_per_episode.values(),
        #             *num_frames_per_episode.values()))

        # Print worst episodes

        n = len(logs["num_frames_per_episode"])
        # if n > 0:
        #     print("\n{} worst episodes:".format(n))

        #     indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        #     for i in indexes[:n]:
        #         print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
            
        #get the success rate at each time step:
        success_rate = []
        #longest episode:
        longest_episode = int(max(logs["num_frames_per_episode"]))
        max_episode = 200
        for i in range(1, max_episode):
            succseeded_in_i_steps = sum([1 for j in range(n) if logs["num_frames_per_episode"][j] < i])
            success_rate.append(succseeded_in_i_steps/n)
            wandb.log({f"Evaluation/{eval_tasks[task+1]}_success_rate": success_rate[-1],f"{eval_tasks[task+1]}_eval_step": i})
        # wandb.log({"avg_return": np.mean(logs["return_per_episode"]), "eval_step": longest_episode})

if __name__ == "__main__":
    main()



