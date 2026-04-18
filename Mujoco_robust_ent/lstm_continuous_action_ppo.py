# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from normalize import NormalizeObservation, NormalizeReward
from torch.distributions.normal import Normal
import argparse
import mujoco_local
import utils
import torch.nn.functional as F
def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm specific arguments
    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=6, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", default="TO_BE_FILLED", help="the wandb's project name")
    parser.add_argument("--wandb_entity", default="TO_BE_FILLED", help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=bool, default=False, help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save_model", type=bool, default=True, help="whether to save model into the `runs_new_reccurent/{run_name}` folder")
    parser.add_argument("--save_dir", type=str, default="runs_puck", help="the dir to save the model")
    parser.add_argument("--upload_model", type=bool, default=False, help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf_entity", default="", help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--env_id", default="CustomPusher-v1", help="the id of the environment")
    parser.add_argument("--control_cost", type=int,default=0, help="coefficient of the control cost")
    parser.add_argument("--total_timesteps", type=int, default=5000000, help="total timesteps of the experiments")
    parser.add_argument("--learning_rate", type=float, default=6e-5, help="the learning rate of the optimizer")
    parser.add_argument("--num_envs", type=int, default=32, help="the number of parallel game environments")
    parser.add_argument("--num_steps", type=int, default=100, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal_lr", type=bool, default=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=bool, default=True, help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.08, help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=bool, default=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--vf_coef", type=float, default=0.3, help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--beta", type=float, default=0.0, help="coefficient of the state entropy")
    parser.add_argument("--ent_coef_decay", type=bool, default=False, help="decay rate of the action entropy")
    parser.add_argument("--beta_discount", type=float, default=0.91, help="decay rate of the state entropy")
    parser.add_argument("--starting_beta",type=float, default=0.0,help="starting beta value for warmup" )
    parser.add_argument("--network_hidden_size", type=int, default=256, help="the size of the hidden layer in the network")
    # to be filled in runtime
    parser.add_argument("--batch_size", type=int, default=0, help="the batch size (computed in runtime)")
    parser.add_argument("--minibatch_size", type=int, default=0, help="the mini-batch size (computed in runtime)")
    parser.add_argument("--num_iterations", type=int, default=0, help="the number of iterations (computed in runtime)")
    args = parser.parse_args()
    args.wandb_group_name = f"action_entropy_{args.ent_coef}__state_entropy_{args.beta}_{args.starting_beta}"
    return args


def make_env(env_id, idx, capture_video, run_name, gamma, control_cost=0,horizon=100):
    def thunk():
        is_ant = "ANT" in env_id.upper()
        gym_kwargs = {"render_mode": "rgb_array"} if (capture_video and idx == 0) else {}
        if is_ant:
            gym_kwargs["exclude_current_positions_from_observation"] = False
        else:
            gym_kwargs["reward_control_weight"] = control_cost
        env = gym.make(env_id, **gym_kwargs)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env=gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
       
        
        return env

    return thunk



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env, hidden_size=256):
        super().__init__()

        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(),256)),
            nn.ReLU(),
            layer_init(nn.Linear(256,256)),
            nn.ReLU())
            
        self.lstm = nn.LSTM(256, 256)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param,0.5)

        self.fc2 = layer_init(nn.Linear(hidden_size, hidden_size))
        self.fc_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, np.prod(env.single_action_space.shape)), std=0.01))
        
        self.fc_logstd = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, np.prod(env.single_action_space.shape)), std=0.01))
        
        self.fc_logstd[-1].weight.data.fill_(0.2)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.LOG_STD_MAX = 1.2
        self.LOG_STD_MIN = -5

    def forward(self, hidden):
        x = F.relu(self.fc2(hidden))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats 
        return mean, log_std



    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        value = self.critic(hidden)
        mean, log_std = self(hidden)
        std = log_std.exp()
        probs = torch.distributions.Normal(mean, std)
        if action is None:
            x_t = probs.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = probs.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1)
        else:
            y_t = (action - self.action_bias) / self.action_scale
            x_t = torch.atanh(torch.clamp(y_t, -0.999, 0.999))  # inverse tanh
            log_prob = probs.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1)

        entropy = probs.entropy().sum(1)
        return action, log_prob, entropy, value, lstm_state
    
if __name__ == "__main__":
    args = parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.seed}__alpha_{args.ent_coef}_beta_{args.beta}_starting_beta_{args.starting_beta}"
    if args.save_model and not os.path.exists(f"{args.save_dir}"):
            os.makedirs(f"{args.save_dir}", exist_ok=True)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=False,
        )
    writer = SummaryWriter(f"{args.save_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma,horizon=args.num_steps,control_cost=args.control_cost) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    unnormed_obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    discount =utils.discount(args.beta_discount,rewards.shape).to(device)
    beta = args.starting_beta
    ent_coef = args.ent_coef
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs,reset_info  = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    unnormed_ob = torch.Tensor(np.stack(reset_info['unnormalized'])).to(device)
    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.starting_beta != args.beta:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            beta = args.beta + (args.starting_beta - args.beta) / 2 * (1 + np.cos(np.pi * (1-frac)))
        
        if args.ent_coef_decay:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            ent_coef_now = frac * args.ent_coef
            ent_coef = ent_coef_now

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            unnormed_obs[step] = unnormed_ob
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            unnormed_ob = torch.Tensor(np.stack(infos['unnormalized'])).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        if "reward_dist" in info:
                            writer.add_scalar("charts/goal_dist", info["reward_dist"], global_step)



        # compute the state entropies:
        is_ant_env = "ANT" in args.env_id.upper()
        state_feats = unnormed_obs[:, :, 0:2] if is_ant_env else unnormed_obs[:, :, 17:19]
        state_entropies = torch.zeros((args.num_steps, args.num_envs)).to(device)
        #looping over the envs to save memory, it bearly affects compute time
        for i in range(args.num_envs):
            state_entropy = utils.compute_state_entropy(src_feats=state_feats[:,i],tgt_feats=state_feats,batch=True,calc_steps=10)
            state_entropies[:,i] = state_entropy
        

        #copute the decayed state entropies:
        state_entropies = state_entropies * discount
        # normalize the state entropi   es as are the rewards
        normalize_var = torch.Tensor(infos['return_normalize_var']).to(device)
        normed_state_entropies = state_entropies/ torch.sqrt(normalize_var + 1e-8)
       
        
        rewards += beta *normed_state_entropies

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/state_entropy", state_entropies.mean(dim=1).sum(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"{args.save_dir}/{run_name}/agent.pth"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        #save the normalization stats:
        normalize_count = infos['normalize_count']
        normalize_mean = infos['normalize_mean']
        normalize_var = infos['normalize_var']
        np.savez(model_path.replace(".pth", "_normalize.npz"), normalize_count=normalize_count, normalize_mean=normalize_mean, normalize_var=normalize_var)


    envs.close()
    writer.close()

