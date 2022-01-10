from ppo import __version__
import gym
from ppo import PPO


def test_version():
    assert __version__ == "0.1.0"

def test_ppo():
    env = gym.make("Pong-v0")
    ppo = PPO(env, 1)