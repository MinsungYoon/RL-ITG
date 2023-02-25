import torch
import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity, BN=False, DR=False):
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes)-2:
            layers += [nn.Linear(sizes[j], sizes[j+1])]
            if BN == True:
                layers += [nn.BatchNorm1d(sizes[j+1])]
            layers += [activation()]
            if DR == True:
                layers += [nn.Dropout(p=0.3)]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), output_activation()]
    return nn.Sequential(*layers)


########### Actor ###########
class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR):
        super(MLPActor, self).__init__()
        pi_sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.pi = mlp(pi_sizes, activation, output_activation=nn.Tanh, BN=BN, DR=DR)
        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

    def forward(self, obs):
        out = self.pi(obs)
        # Return output from network scaled to action space limits.
        if obs.is_cuda:
            return (self.act_std.cuda() * out) + self.act_mean.cuda()
        else:
            return (self.act_std * out) + self.act_mean

########### Q ###########
class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPQFunction, self).__init__()
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1], activation, output_activation=nn.Identity)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

########### Actor Critic ###########
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_ll, act_ul, BN, DR, Boost_Q, hidden_sizes=None,
                 activation=nn.ReLU):
        super(MLPActorCritic, self).__init__()

        act_std = (act_ul - act_ll)/2
        act_mean = (act_ul + act_ll)/2

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)

        hidden_sizes = [h + Boost_Q for h in hidden_sizes]
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
