import gym
import torch
from torch import nn
import pytorch_lightning as pl

import ppo

class PPO(pl.LightningModule):
    def __init__(self, env: gym.Env, std_init) -> None:
        super().__init__()
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.net = ppo.ActorCritic(self.obs_dim, self.action_dim, std_init)

    def forward(self, obs):
        return self.net(obs)
        
        
if __name__ == "__main__":
    env = gym.make("Pong-v0")
    ppo = PPO(env, 1)



    

    