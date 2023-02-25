### Run using 'python3'

import os
import numpy as np
import rospy
import torch
import torch.nn as nn

from time import perf_counter
def timer(f,*args):
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))

mode = 'BC'

# # ================ [load sac neural interpolator] ================
# from algos.sac import core
# cur_dir_path = os.path.dirname(os.path.abspath(__file__))
# log_path = 'logs/backup/WRwd2_10_0.05_2.5_0.1_5_10_20_5_dth10_rth10__ep_1000_plr_0.0001_qlr_0.0003_pwd_1e-05_qwd_0.001_gclip_1.0_hids_[512, 256, 128]_BQ_1024_BN_False_DR_False'
# log_dir = os.path.join(cur_dir_path, log_path)
# pi_state_dict = torch.load(os.path.join(log_dir, 'best_model.pt'))
#
# obs_dim = 35
# act_dim = 7
# act_ll = np.array(rospy.get_param("/robot/vel_ll"))
# act_ul = np.array(rospy.get_param("/robot/vel_ul"))
#
# BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
# DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : ])
# Boost_Q = eval(log_path[ log_path.find('BQ')+len('BQ')+1 : log_path.find('BN')-1])
# hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BQ')-1])
#
# ac = core.MLPActorCritic(obs_dim, act_dim, act_ll, act_ul, BN, DR, Boost_Q, hidden_sizes)
# ac.load_state_dict(pi_state_dict)

# Behavior cloning =================================================================================================
if mode == 'BC':
    from src.pirl.script.imitation_learning import core
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    log_path = './logs/NewR3_T0_ep_10000000_bs_4096_plr_7e-05_qlr_0.0001_pwd_0.0001_qwd_0.0001_gclip_10.0_hids_[1024, 1024, 1024]_BQ_0_BN_False_DR_False_AET_True_elr_0.0003_TEnt_5_AReg_3_SFS_False'
    log_dir = os.path.join(cur_dir_path, log_path)
    state_dict = torch.load(os.path.join(log_dir, 'last_model.pt'))

    save_model_name = "./bc_basic_model.pt"


    obs_dim = 133
    act_dim = 7

    is_Actll = log_path[ log_path.find('Actll')+len('Actll')+1 : ] == "vel"
    if is_Actll:
        act_ll = np.array([-1.256, -1.454, -1.571, -1.521, -1.571, -2.268, -2.268])
        act_ul = np.array([1.256, 1.454, 1.571, 1.521, 1.571, 2.268, 2.268])
    else:
        act_ll = np.array([-6.28]*7)
        act_ul = np.array([6.28]*7)
    act_mean = (act_ul + act_ll)/2
    act_std = (act_ul - act_ll)/2

    BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
    DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : log_path.find('Actll')-1])
    hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BN')-1])
    activation = nn.ELU

    pi = core.SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)
    pi.load_state_dict(state_dict)
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
loaded = torch.jit.load('./serialized_model.pt')
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

