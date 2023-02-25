### Run using 'python3'

import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


from time import perf_counter
def timer(f,*args):
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))


import serialize_model_core as core
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
# log_path = '/data/pirl_network/logs/FREEALL/R0-0-0-10P10-1-3_free_Imitation_Conf+LinkPoses_T1_ep_100000000_bs_4096_plr_7e-05_qlr_0.0001_pwd_0.0001_qwd_0.0001_gclip_10.0_hids_[1024, 1024, 1024]_BQ_0_BN_False_DR_False_AET_True_elr_0.0001_TEnt_3_AReg_3_SFS_False'
log_path = '/data/pirl_network/logs/FREEALL/JACO_free_Task+NullSpaceImitation_Conf+LinkPoses_T1_ep_100000000_bs_4096_plr_7e-05_qlr_0.0001_pwd_0.0001_qwd_0.0001_gclip_10.0_hids_[1024, 1024, 1024]_BQ_0_BN_False_DR_False_AET_True_elr_0.0001_TEnt_3_AReg_3_SFS_False_ZDim_0_PO_0.0_ZS_None_RF_False'
pt_name = 'best_model.pt'
state_dict = torch.load(os.path.join(log_path, pt_name))

obsType = "Conf+LinkPoses"
T = 1
memo = 'JACOFREE'
save_model_name = "./rl_{}_T{}_model_{}_{}.pt".format(obsType, T, memo, pt_name.split("_")[0])

obsDim = 0
if obsType == "onlyConf":
    obsDim += 7
elif obsType == "Conf+LinkPoses":
    obsDim += 7 + 9 * 8
elif obsType == "Conf+EE":
    obsDim += 7 + 9
elif obsType == "onlyLinkPoses":
    obsDim += 9 * 8
obsDim += 9 * T

if memo.find("OBS") != -1:
    obsDim += eval(log_path[log_path.find("ZDim_")+len("ZDim_"):log_path.find("_PO_")])

obs_dim = obsDim
act_dim = 7

act_ll = np.array([-0.26, -0.26, -0.26, -0.26, -0.26, -0.26, -0.26])
act_ul = np.array([0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26])
act_mean = (act_ul + act_ll)/2
act_std = (act_ul - act_ll)/2

BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : log_path.find('AET')-1])
hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BQ')-1])
activation = nn.ELU

pi = core.SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)

pi_state_dict = OrderedDict()
for name, value in state_dict.items():
    if name.find('pi') >= 0:
        pi_state_dict[name[3:]] = value
pi.load_state_dict(pi_state_dict)
pi.eval()

print("Model has been loaded.")
print("[1]======================")
from serialize_model_core import PirlPolicy
pirl_pi = PirlPolicy(obs_dim, act_dim, act_ll, act_ul, BN, DR, hidden_sizes)

pirl_pi.pi_backdone = pi.pi_backdone
pirl_pi.pi_mu = pi.pi_mu
pirl_pi.pi_log_std = pi.pi_log_std

x = torch.randn(obs_dim)
print(pirl_pi(x, True))
print(pirl_pi(x, False))
print(pirl_pi(x, False))
elt = np.mean([timer(pirl_pi,x,False) for _ in range(1000)])
print("[TIME] {} ms".format(elt))

# pirl_pi = pirl_pi.cuda()
print("[2]======================")
scriptmodel = torch.jit.script(pirl_pi)
print(scriptmodel)
print(scriptmodel.code)
print(scriptmodel(x, True))
print(scriptmodel(x, False))
print(scriptmodel(x, False))
elt = np.mean([timer(scriptmodel,x,False) for _ in range(1000)])
print("[TIME] {} ms".format(elt))
scriptmodel.save(save_model_name)

print("[3]======================")
loaded = torch.jit.load(save_model_name)
print(loaded)
print(loaded.code)
print(loaded(x, True))
print(loaded(x, False))
print(loaded(x, False))
elt = np.mean([timer(loaded,x,False) for _ in range(1000)])
print("[TIME] {} ms".format(elt))

# import ipdb
# ipdb.set_trace()


#
# class MyModule(torch.nn.Module):
#     def __init__(self, N, M):
#         super(MyModule, self).__init__()
#         self.weight = nn.Linear(3,3)
#         self.i = 100
#
#     def forward(self, input):
#         if input.sum() > 0:
#             output = self.weight(input)
#         else:
#             output = self.weight(input) + input + self.i
#         return output
#
# my_module = MyModule(10,20)
# print(my_module)
# sm = torch.jit.script(my_module)
# print(sm)
# print(sm.code)

