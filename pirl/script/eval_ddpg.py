#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import os
import rospy
from envs.PirlEnv import PirlEnv
from algos.ddpg import core
import time
import torch

rospy.init_node("eval_main")

cur_dir_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'logs/floor/sac/Dw7Rw1S5R3C5_RndSG__ep_5000_plr_3e-05_qlr_7e-05_pwd_1e-05_qwd_0.0007_gclip_1.0_hids_[256, 128]_BQ_256_BN_False_DR_False'
log_dir = os.path.join(cur_dir_path, log_path)
pi_state_dict = torch.load(os.path.join(log_dir, 'best_model.pt'))

env = PirlEnv(demo=True, is_test=True)

obs_dim = env.observation_space_shape[0]
act_dim = env.action_space_shape[0]

BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : ])
Boost_Q = eval(log_path[ log_path.find('BQ')+len('BQ')+1 : log_path.find('BN')-1])
hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BQ')-1])

ac = core.MLPActorCritic(obs_dim, act_dim, env.act_ll, env.act_ul, BN, DR, Boost_Q, hidden_sizes)
ac.load_state_dict(pi_state_dict)

print("Model has been loaded.")

def get_action(o):
    o = torch.as_tensor(o, dtype=torch.float32)
    if o.dim() == 1:
        o = o.unsqueeze(0)
    a = ac.act(o)[0]
    return a

num_test_episodes = 10

test_log_ep_ret, test_log_ep_len = [], []
n_suc, n_col, n_timeout = 0.0, 0.0, 0.0
for i in range(num_test_episodes):
    # rollout
    o, d, ep_ret, ep_len = env.reset(), False, 0.0, 0.0
    while not d:
        # Take deterministic actions at test time
        a = get_action(o)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1
        print("[TEST] Reward: {:4.3f} S:{:4} | C:{:4} | TO:{:4}".format(
            r, info['is_suc'], info['is_col'], info['is_timeout']
        ))
    print("\n")
    env.visualize_demo()
    n_suc       += info['is_suc']
    n_col       += info['is_col']
    n_timeout   += info['is_timeout']
    test_log_ep_ret.append(ep_ret)
    test_log_ep_len.append(ep_len)
    rospy.sleep(10)
print("[TEST] Nsuc: {} Ncol: {} Ntimeout: {} | EpRet: {} EpLen: {}".format(
    n_suc, n_col, n_timeout, sum(test_log_ep_ret)/len(test_log_ep_ret), sum(test_log_ep_len)/len(test_log_ep_len)
))
