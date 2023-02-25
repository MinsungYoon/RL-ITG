import numpy as np
import math

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

###################################################################################################################

class NormalMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR):
        super(NormalMLP, self).__init__()
        pi_backbone_dim = [obs_dim] + hidden_sizes + [act_dim]
        self.pi_backdone = mlp(pi_backbone_dim, activation, output_activation=nn.Tanh, BN=BN, DR=DR)
        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

    def forward(self, obs, deterministic=False, with_logprob=True):
        pi_action = self.pi_backdone(obs)
        pi_action = torch.tanh(pi_action)

        if obs.is_cuda:
            return (self.act_std.cuda() * pi_action) + self.act_mean.cuda()
        else:
            return (self.act_std * pi_action) + self.act_mean

###################################################################################################################

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


class SquashedGaussianMLPActor_ver2(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR):
        super(SquashedGaussianMLPActor_ver2, self).__init__()
        self.fc_l0 = nn.Linear(72, 128)
        self.fc_l1 = nn.Linear(128, 32)

        self.fc_t0 = nn.Linear(54, 128)
        self.fc_t1 = nn.Linear(128, 32)

        self.fc_b0 = nn.Linear(7+32+32, 1024)
        self.pi_mu = nn.Linear(1024, act_dim)
        self.pi_log_std = nn.Linear(1024, act_dim)

        self.act_mean = torch.Tensor(act_mean)
        self.act_std = torch.Tensor(act_std)

        self.LOG_STD_MAX = 2   # max_std: e^2 == 7.38
        self.LOG_STD_MIN = -20 # min_std: e^(-20) == 2.06e-9

    def forward(self, obs, deterministic=False, with_logprob=True):

        x_link = F.elu(self.fc_l0(obs[..., 7:7+72]))
        x_link = F.elu(self.fc_l1(x_link))

        x_target = F.elu(self.fc_t0(obs[..., 7+72:]))
        x_target = F.elu(self.fc_t1(x_target))

        x = torch.cat((obs[..., :7], x_link, x_target), dim=-1)

        mu = self.pi_mu(x)
        log_std = self.pi_log_std(x)
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

###################################################################################################################

class MotionPredictionNN(nn.Module):
    def __init__(self, n_input, n_output, n_expert_weights, h=[1024, 1024], drop_prob=0.2):
        super(MotionPredictionNN, self).__init__()
        self.n_expert_weights = n_expert_weights
        self.n_input = n_input
        self.n_output = n_output
        self.h = h
        self.expert_weights_fc0 = nn.Parameter(torch.Tensor(n_expert_weights, h[0], n_input))
        self.expert_bias_fc0    = nn.Parameter(torch.Tensor(n_expert_weights, h[0]))
        self.expert_weights_fc1 = nn.Parameter(torch.Tensor(n_expert_weights, h[1], h[0]))
        self.expert_bias_fc1    = nn.Parameter(torch.Tensor(n_expert_weights, h[1]))
        self.expert_weights_fc2 = nn.Parameter(torch.Tensor(n_expert_weights, n_output, h[1]))
        self.expert_bias_fc2    = nn.Parameter(torch.Tensor(n_expert_weights, n_output))
        self.reset_parameters()
        self.drop_prob = drop_prob

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.expert_weights_fc0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_weights_fc1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_weights_fc2, a=math.sqrt(5))
        nn.init.zeros_(self.expert_bias_fc0)
        nn.init.zeros_(self.expert_bias_fc1)
        nn.init.zeros_(self.expert_bias_fc2)

    def forward(self, x, BC):
        W0, B0, W1, B1, W2, B2 = self.blend(BC)

        x = F.dropout(x, self.drop_prob, self.training)
        x = torch.baddbmm(B0.unsqueeze(2), W0, x.unsqueeze(2))
        x = F.elu(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = torch.baddbmm(B1.unsqueeze(2), W1, x)
        x = F.elu(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = torch.baddbmm(B2.unsqueeze(2), W2, x)
        x = x.squeeze(2)
        return x

    def blend(self, BC):
        BC_w = BC.unsqueeze(2).unsqueeze(2) # (B, 4, 1, 1)
        BC_b = BC.unsqueeze(2)

        W0 = torch.sum(BC_w * self.expert_weights_fc0.unsqueeze(0), dim=1) # (B, 4, 1, 1)*(1, 4, 1024, 133) = (B, 4, 1024, 133) -sum-> (B, 1024, 133)
        B0 = torch.sum(BC_b * self.expert_bias_fc0.unsqueeze(0), dim=1)
        W1 = torch.sum(BC_w * self.expert_weights_fc1.unsqueeze(0), dim=1)
        B1 = torch.sum(BC_b * self.expert_bias_fc1.unsqueeze(0), dim=1)
        W2 = torch.sum(BC_w * self.expert_weights_fc2.unsqueeze(0), dim=1)
        B2 = torch.sum(BC_b * self.expert_bias_fc2.unsqueeze(0), dim=1)
        return W0, B0, W1, B1, W2, B2


class GatingNN(nn.Module):
    def __init__(self, n_input, n_expert_weights, hg=[32, 32], drop_prob=0.0):
        super(GatingNN, self).__init__()
        self.fc0 = nn.Linear(n_input, hg[0])
        self.fc1 = nn.Linear(hg[0], hg[1])
        self.fc2 = nn.Linear(hg[1], n_expert_weights)
        self.drop_prob = drop_prob

    def forward(self, x):
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.elu(self.fc0(x))
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.softmax(self.fc2(x), dim=1)
        return x

class MANN(nn.Module): # mode adaptive neural network
    def __init__(self, index_gating, n_expert_weights=4, hg=[256, 64],
                 n_input_motion=133, n_output_motion=7, h=[1024, 512],
                 drop_prob_gat=0.0, drop_prob_mot=0.2):
        super(MANN, self).__init__()
        self.index_gating = index_gating
        n_input_gating = self.index_gating.shape[0]
        self.gatingNN = GatingNN(n_input_gating, n_expert_weights, hg, drop_prob_gat)
        self.motionNN = MotionPredictionNN(n_input_motion, n_output_motion, n_expert_weights, h, drop_prob_mot)

    def forward(self, x):
        in_gating = x[..., self.index_gating]
        BC = self.gatingNN(in_gating) # BC: blend coefficient
        return self.motionNN(x, BC)

# def count_vars(model):
#     print("[Model Param] #: {}".format(sum([np.prod(p.shape) for p in model.parameters()])))
#
# def show_named_params(model):
#     for n, i in model.named_parameters():
#         print("[PARAMS] {}: \n{}\n".format(n, i))
# def show_named_buffers(model):
#     for n, i in model.named_buffers():
#         print("[BUFFERS] {}: \n{}\n".format(n, i))
#
# index_gating = np.array([0,1,2,3,4,5,6])
#
# mann = MANN(index_gating, n_expert_weights=4)
# show_named_params(mann)
# show_named_buffers(mann)
# count_vars(mann)
#
# rnd_x = torch.randn(5,133)
# mann(rnd_x)