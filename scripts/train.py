import torch
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from models import CNNPolicy, Discriminator
from envs import DepthRefinementEnv
from utils.data_loader import DemoDataset

def train():
    dataset = DemoDataset()
    env = DepthRefinementEnv(dataset)

    policy = CNNPolicy().cuda()
    discriminator = Discriminator().cuda()

    trainer = onpolicy_trainer(
        policy,
        Collector(policy, env, VectorReplayBuffer(100000)),
        max_epoch=100,
        step_per_epoch=1000,
        save_fn=lambda: torch.save(policy.state_dict(), "checkpoint.pth")
    )
    print("Training completed!")
