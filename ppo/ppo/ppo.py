import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ppo import ActorCritic


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """

    def _init():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed + rank)
        return env

    torch.manual_seed(seed)
    return _init


class PPO:
    def __init__(
        self,
        env_id: str,
        num_workers: int,
        ep_len: int,
        minibatch_size: int,
        gamma: int,
        lr: int,
        epsilon: int,
        norm_adv: int,
        max_grad_norm: int,
        clip_vloss: int,
        device: str,
    ):
        self.env_id = env_id
        self.num_workers = num_workers
        self.ep_len = ep_len
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.norm_adv = norm_adv
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss
        self.device = device

        self.envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_workers)])

        obs_dim = self.envs.observation_space.shape[0]
        action_dim = self.envs.action_space.shape[0]
        self.agent = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.writer = SummaryWriter()

    def train(self):
        N = self.num_workers
        T = self.ep_len
        gamma = self.gamma
        K = self.epochs

        minibatch_size = self.minibatch_size
        batch_size = int(N * T)

        global_step = 0
        for _ in tqdm(range(self.rollout_iterations)):
            # Rollout
            reward_batch = torch.zeros((T, N)).to(self.device)
            done_batch = torch.zeros((T, N)).to(self.device)
            value_batch = torch.zeros((T, N)).to(self.device)

            obs_batch = torch.zeros((T, N) + self.envs.observation_space.shape).to(
                self.device
            )
            action_batch = torch.zeros((T, N) + self.envs.action_space.shape).to(
                self.device
            )
            logprob_batch = torch.zeros((T, N)).to(self.device)

            obs = self.envs.reset()
            for t in range(T):
                global_step += 1 * self.envs.num_envs
                with torch.no_grad():
                    action, logprobs, values = self.agent(
                        torch.from_numpy(obs).to(self.device)
                    )

                self.envs.step_async(action.cpu().numpy())

                obs_batch[t] = torch.from_numpy(obs).to(self.device)
                action_batch[t] = action
                logprob_batch[t] = logprobs
                value_batch[t] = values.flatten()

                obs, rewards, dones, infos = self.envs.step_wait()

                reward_batch[t] = torch.from_numpy(rewards).to(self.device)
                done_batch[t] = torch.from_numpy(dones).to(self.device)

                for item in infos:
                    if "episode" in item.keys():
                        self.writer.add_scalar(
                            "charts/episodic_return", item["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", item["episode"]["l"], global_step
                        )
                        break

            # Advantage calculation
            with torch.no_grad():
                _, _, next_value = self.agent(torch.from_numpy(obs).to(self.device))
                next_value = values.view(1, -1)

                return_batch = torch.zeros_like(reward_batch).to(self.device)
                for t in reversed(range(T)):
                    if t == T - 1:
                        nextnonterminal = 1.0 - dones
                        nextnonterminal = torch.from_numpy(nextnonterminal).to(
                            self.device
                        )
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - done_batch[t + 1]
                        next_return = return_batch[t + 1]

                    return_batch[t] = (
                        reward_batch[t] + gamma * nextnonterminal * next_return
                    )
                advantage_batch = return_batch - value_batch

            # Optimization
            obs_batch = obs_batch.view((-1,) + self.envs.observation_space.shape)
            logprob_batch = logprob_batch.view(-1)
            value_batch = value_batch.view(-1)
            advantage_batch = advantage_batch.view(-1)
            return_batch = return_batch.view(-1)

            b_inds = np.arange(batch_size)
            for _ in range(K):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprobs, newvalues = self.agent(obs_batch[mb_inds])

                    ratio = (newlogprobs - logprob_batch[mb_inds]).exp()

                    mb_advantages = advantage_batch[mb_inds]

                    # Normalise advantage minibatch
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Clipped loss
                    loss1 = -mb_advantages * ratio
                    loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.epsilon, 1 + self.epsilon
                    )
                    clip_loss = torch.max(loss1, loss2).mean()

                    # Value loss
                    newvalues = newvalues.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalues - return_batch[mb_inds]) ** 2
                        v_clipped = value_batch[mb_inds] + torch.clamp(
                            newvalues - value_batch[mb_inds],
                            -self.epsilon,
                            self.epsilon,
                        )
                        v_loss_clipped = (v_clipped - return_batch[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()

                    else:
                        v_loss = 0.5 * ((newvalues - value_batch[mb_inds]) ** 2).mean()

                    # Total loss, switch signs because of gradient descent (as opposed to gradient ascent)
                    loss = clip_loss + v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

            self.writer.add_scalar("losses/clip_loss", clip_loss.item(), global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/total_loss", loss.item(), global_step)
