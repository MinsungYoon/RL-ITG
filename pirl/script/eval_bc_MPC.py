#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import os
import rospy
import time
import torch
import numpy as np
import torch.nn as nn

from envs.PirlTrajEnv_MPC import PirlTrajEnv_Fetch
from imitation_learning_MPC import core

rospy.init_node("eval_main")

cur_dir_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'imitation_learning_MPC/log/SGMLP_ep_10000_lr_0.0003_bs_8192_step_500_wd_7e-06_gclip_0.0_hids_[1024, 1024, 1024, 1024]_BN_False_DR_False_Actll_vel'
model_path = log_path + '/bc_best_model.pt'
log_dir = os.path.join(cur_dir_path, model_path)
state_dict = torch.load(log_dir)

env = PirlTrajEnv_Fetch()

obs_dim = env.observation_space_shape[0]
act_dim = env.action_space_shape[0]

is_Actll = log_path[ log_path.find('Actll')+len('Actll')+1 : ] == "vel"
if is_Actll:
    act_ll = np.array([-1.256, -1.454, -1.571, -1.521, -1.571, -2.268, -2.268]*6)
    act_ul = np.array([1.256, 1.454, 1.571, 1.521, 1.571, 2.268, 2.268]*6)
else:
    act_ll = np.array([-6.28]*7*6)
    act_ul = np.array([6.28]*7*6)
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

def get_action(o): # default: stocastic action
    return pi(torch.as_tensor(o, dtype=torch.float32), deterministic=False, with_logprob=False)

num_test_episodes = 10

test_log_ep_ret, test_log_ep_len = [], []
n_suc, n_col, n_timeout = 0.0, 0.0, 0.0
for i in range(num_test_episodes):
    # rollout
    o, d, ep_ret, ep_len = env.reset(mode='test', set_from_start=True), False, 0.0, 0.0
    while not d:
        # Take deterministic actions at test time
        # print(o)
        a = get_action(o)[0].detach().numpy()
        # a = env.demo_configs[env.timestep+1] - env.demo_configs[env.timestep]
        # for i in range(len(a)):
        #     if i in [2,4,6]:
        #         if a[i]>np.pi:
        #             a[i] -= 2*np.pi
        #         elif a[i] < -np.pi:
        #             a[i] += 2*np.pi
        # print("A: {:3.3f} {:3.3f} {:3.3f} {:3.3f} {:3.3f} {:3.3f} {:3.3f}".format(
        #     a[0], a[1], a[2], a[3], a[4], a[5], a[6]
        # ))
        o, r, d, info = env.step(a)
        d = info["is_timeout"]
        # if d:
        #     print(info)
        # print("[INFO] L: {}, DS:{:2} | RS:{:2} | C:{:2} | JR:{:2} | TO:{:2}, ES:{:2}".format(
        #     ep_len, info['is_dist_suc'], info['is_rot_suc'], info['is_col'], info['is_jlimit'], info['is_timeout'], info['is_early_stop']
        # ))
        ep_ret += r
        ep_len += 1

    env.visualize_solution()
    print("[TEST] Solution")
    raw_input("Enter for new experiment.")

    env.visualize_demo()
    print("[TEST] Demo")
    raw_input("Enter for showing test result.")

    test_log_ep_ret.append(ep_ret)
    test_log_ep_len.append(ep_len)

print("[TEST] EpRet: {} EpLen: {}".format(
    sum(test_log_ep_ret)/len(test_log_ep_ret), sum(test_log_ep_len)/len(test_log_ep_len)
))
