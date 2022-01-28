import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor_mean = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Covariance for multivariate normal policy
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self, obs):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        action = probs.sample()
        value = self.critic(obs)

        return action, probs.log_prob(action).sum(1), value
