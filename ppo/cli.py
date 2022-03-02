import argparse

import torch

from ppo import PPO

def main():
    parser = argparse.ArgumentParser(description="Train PPO")
    parser.add_argument(
        "--env_id", type=str, default="LunarLanderContinuous-v2", help="Gym env id"
    )
    parser.add_argument("--cuda", type=bool, default=True, help="PyTorch cuda")
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of parellel environments"
    )
    parser.add_argument("--ep_len", type=int, default=2048, help="Episode length")
    parser.add_argument("--rollout_iterations", type=int, default=100, help="Number of rollout iterations")
    parser.add_argument("--epochs", type=int, default=10, help="Number of gradient ascent epochs")
    parser.add_argument(
        "--minibatch_size", type=int, default=64, help="Mini-batch size"
    )
    parser.add_argument("--gamma", type=int, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=int, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon", type=int, default=0.2, help="Clip coefficient")
    parser.add_argument(
        "--norm_adv", type=bool, default=True, help="Normalise advantage"
    )
    parser.add_argument(
        "--max_grad_norm", type=int, default=1.0, help="Gradient Normalisation"
    )
    parser.add_argument("--clip_vloss", type=bool, default=True, help="Clip value loss")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ppo = PPO(
        env_id=args.env_id,
        num_workers=args.num_workers,
        ep_len=args.ep_len,
        rollout_iterations=args.rollout_iterations,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=args.epsilon,
        norm_adv=args.norm_adv,
        max_grad_norm=args.max_grad_norm,
        clip_vloss=args.clip_vloss,
        device=device,
    )
    ppo.train()

if __name__ == "__main__":
    main()