### Run using 'python3'

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

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

#################################################################################################################
class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR):
        super(SquashedGaussianMLPActor, self).__init__()
        pi_backbone_dim = [obs_dim] + hidden_sizes
        self.pi_backdone = mlp(pi_backbone_dim, activation, activation, BN=BN, DR=DR)
        self.pi_mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.pi_log_std = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

        self.LOG_STD_MAX = 2   # max_std: e^2 == 7.38
        self.LOG_STD_MIN = -20 # min_std: e^(-20) == 2.06e-9

    def forward(self, obs, deterministic=False, with_logprob=True):
        backbone_out = self.pi_backdone(obs)
        mu = self.pi_mu(backbone_out)
        log_std = self.pi_log_std(backbone_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_dist = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_dist.rsample() # rsample: reparameterization_sample

        if with_logprob:
            logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)

        if obs.is_cuda:
            return (self.act_std.cuda() * pi_action) + self.act_mean.cuda(), logp_pi, mu, std
        else:
            return (self.act_std * pi_action) + self.act_mean, logp_pi, mu, std


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPQFunction, self).__init__()
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1], activation, output_activation=nn.Identity)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_ll, act_ul, BN, DR, Boost_Q, hidden_sizes=None,
                 activation=nn.ELU):
        super(MLPActorCritic, self).__init__()

        act_std = (act_ul - act_ll)/2
        act_mean = (act_ul + act_ll)/2

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)

        hidden_sizes = [h + Boost_Q for h in hidden_sizes]
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False): # default: stocastic action
        with torch.no_grad():
            a, _, _, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
            return a.numpy()

#################################################################################################################

# for exporting to C++

from torch import tensor as Tensor
class PirlPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, act_ll, act_ul, BN, DR, hidden_sizes=None,
                 activation=nn.ReLU):
        super(PirlPolicy, self).__init__()

        self.act_std = torch.Tensor((act_ul - act_ll)/2)
        self.act_mean = torch.Tensor((act_ul + act_ll)/2)

        pi_backbone_dim = [obs_dim] + hidden_sizes
        self.pi_backdone = mlp(pi_backbone_dim, activation, activation, BN=BN, DR=DR)
        self.pi_mu = nn.Linear(hidden_sizes[-1], act_dim)
        self.pi_log_std = nn.Linear(hidden_sizes[-1], act_dim)
        self.LOG_STD_MAX = 2   # max_std: e^2 == 7.38
        self.LOG_STD_MIN = -20 # min_std: e^(-20) == 2.06e-9

    def forward(self, obs: Tensor, deterministic: bool) -> Tensor:
        backbone_out = self.pi_backdone(obs)
        mu = self.pi_mu(backbone_out)

        if deterministic:
            pi_action = mu
        else:
            log_std = self.pi_log_std(backbone_out)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            pi_action = eps * std + mu

        pi_action = torch.tanh(pi_action)
        return (self.act_std * pi_action) + self.act_mean
