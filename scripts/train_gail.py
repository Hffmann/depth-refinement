from tianshou.policy import GAILPolicy
from tianshou.trainer import onpolicy_trainer

def train_refinement_policy(dataset, zerodepth):
    env = DepthRefinementEnv(dataset, zerodepth)
    policy = CNNPolicy().cuda()

    # GAIL setup
    policy = GAILPolicy(
        policy,
        expert_buffer=dataset.expert_buffer,
        disc_net=Discriminator().cuda(),
        optim=torch.optim.Adam(policy.parameters(), lr=1e-4)
    )

    # Training
    result = onpolicy_trainer(
        policy,
        Collector(policy, env, VectorReplayBuffer(20000)),
        max_epoch=100,
        step_per_epoch=1000,
    )
    return policy
