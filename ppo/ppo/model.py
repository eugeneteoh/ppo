import torch
from torch import nn
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, std_init):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Covariance for multivariate normal policy
        self.action_vars = torch.full((self.action_dim,), std_init * std_init)
        self.cov = torch.diag(self.action_vars)

    def forward(self, obs):
        means = self.actor(obs)
        policy = MultivariateNormal(means, self.cov)
        action = policy.sample()
        
        value = self.critic(obs)
        
        return action, value
        
if __name__ == "__main__":
    model = ActorCritic(50, 5, 1)
    print(model.cov.shape)

        