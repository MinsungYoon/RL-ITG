#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import os
import rospy
from envs.PirlTrajEnv import PirlTrajEnv_Fetch
from algos.sac import core
import time
import torch

rospy.init_node("eval_main")

cur_dir_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'logs/sac/501__ep_100000_bs_8192_plr_0.0001_qlr_0.001_pwd_1e-05_qwd_0.0001_gclip_7.0_hids_[1024, 1024, 1024]_BQ_0_BN_False_DR_True_AET_True_elr_0.0003_TEnt_5_SFS_False'
log_dir = os.path.join(cur_dir_path, log_path)
state_dict = torch.load(os.path.join(log_dir, 'last_model.pt'))


env = PirlTrajEnv_Fetch()

obs_dim = env.observation_space_shape[0]
act_dim = env.action_space_shape[0]

BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : log_path.find('AET')-1])
Boost_Q = eval(log_path[ log_path.find('BQ')+len('BQ')+1 : log_path.find('BN')-1])
hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BQ')-1])


ac = core.MLPActorCritic(obs_dim, act_dim, env.act_ll, env.act_ul, BN, DR, Boost_Q, hidden_sizes)
ac.pi.load_state_dict(state_dict)

ac.eval()

print("Model has been loaded.")

def get_action(o, deterministic=False): # default: stocastic action
    return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

num_test_episodes = 5

test_log_ep_ret, test_log_ep_len = [], []
n_suc, n_col, n_timeout = 0.0, 0.0, 0.0
for i in range(num_test_episodes):
    # rollout
    o, d, ep_ret, ep_len = env.reset(w=5, p=0, gap=1, mode='train', set_from_start=True), False, 0.0, 0.0
    while not d:
        # Take deterministic actions at test time
        a = get_action(o, deterministic=True)
        # print("A: {:3.3f} {:3.3f} {:3.3f} {:3.3f} {:3.3f} {:3.3f} {:3.3f}".format(
        #     a[0], a[1], a[2], a[3], a[4], a[5], a[6]
        # ))
        o, r, d, info = env.step(a)
        # print("[INFO] L: {}, DS:{:2} | RS:{:2} | C:{:2} | JR:{:2} | TO:{:2}, ES:{:2}".format(
        #     ep_len, info['is_dist_suc'], info['is_rot_suc'], info['is_col'], info['is_jlimit'], info['is_timeout'], info['is_early_stop']
        # ))
        ep_ret += r
        ep_len += 1

    # env.visualize_demo()
    # print("[TEST] Demo")
    # rospy.sleep(5)

    env.visualize_solution()
    print("[TEST] Solution")
    rospy.sleep(5)

    test_log_ep_ret.append(ep_ret)
    test_log_ep_len.append(ep_len)

print("[TEST] EpRet: {} EpLen: {}".format(
    sum(test_log_ep_ret)/len(test_log_ep_ret), sum(test_log_ep_len)/len(test_log_ep_len)
))
