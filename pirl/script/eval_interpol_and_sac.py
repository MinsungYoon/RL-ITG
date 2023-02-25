#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import os
import rospy
import time
import torch
import numpy as np
from envs.PirlEnv import PirlEnv
from algos.sac import core

from pirl_msgs.srv import fk
from pirl_msgs.srv import ik
from pirl_msgs.srv import rndvalconf

from envs.utils import PathVisualizer

from utils.interpolation_utils import path_refine_and_interpolation

def pathwise_fk(config_list):
    pos_list = []
    for conf in config_list:
        res = fk_srv(conf)
        while not res.result:
            res = fk_srv(conf)
        pos_list.append(list(res.fk_result))
    return pos_list

rospy.init_node("traj_test_main")


path_viz = PathVisualizer(is_test=True, is_second_path=True)
rospy.sleep(5)

# ================================================================
# ================ [start and goal setting] ======================
fk_srv = rospy.ServiceProxy('/fk_solver', fk)

rndvalconf_srv = rospy.ServiceProxy('/rndvalconf', rndvalconf)
fk_base_position = rospy.get_param("/robot/fk_base_position")
sample_range = rospy.get_param("/robot/sample_range")
sample_range[0] = sample_range[0] - fk_base_position[0]
sample_range[1] = sample_range[1] - fk_base_position[0]
sample_range[2] = sample_range[2] - fk_base_position[1]
sample_range[3] = sample_range[3] - fk_base_position[1]
sample_range[4] = sample_range[4] - fk_base_position[2]
sample_range[5] = sample_range[5] - fk_base_position[2]

res = rndvalconf_srv(sample_range)
while not res.result:
    res = rndvalconf_srv(sample_range)
start_conf = np.array(res.configuration)
start_pose = np.array(res.ee_pose)
path_viz.pub_arrow(start_pose, color='b')

res = rndvalconf_srv(sample_range)
while not res.result:
    res = rndvalconf_srv(sample_range)
goal_conf = np.array(res.configuration)
goal_pose = np.array(res.ee_pose)
path_viz.pub_arrow(goal_pose, color='r')

# ================================================================
# ======= [naive interpolation in configuration space] ===========
init_conf = []
init_conf.append(start_conf)
init_conf.append(goal_conf)

interpolated_conf = path_refine_and_interpolation(init_conf, N=50, c_joints=[2,4,6])

interpolated_pose_list = pathwise_fk(interpolated_conf)

print("[INFO] interpolated path length: {}".format(len(interpolated_conf)))
path_viz.pub_path(interpolated_conf)
path_viz.pub_eepath_arrow(interpolated_pose_list)
path_viz.pub_eepath_strip(interpolated_pose_list)

for conf in interpolated_conf:
    print("{:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f}".format(conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]))

print("[INFO] stop for 5sec.")
time.sleep(5)
# ================================================================
# ================ [load sac neural interpolator] ================
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'logs/backup/WRwd2_10_0.05_10_0.1_5_5_20_5_dth10_rth10__ep_1000_plr_0.0001_qlr_0.0003_pwd_1e-05_qwd_0.001_gclip_1.0_hids_[512, 256, 128]_BQ_1024_BN_False_DR_False'
log_dir = os.path.join(cur_dir_path, log_path)
pi_state_dict = torch.load(os.path.join(log_dir, 'best_model.pt'))

env = PirlEnv(demo=True, is_test=True, is_second_path=True)

obs_dim = env.observation_space_shape[0]
act_dim = env.action_space_shape[0]

BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : ])
Boost_Q = eval(log_path[ log_path.find('BQ')+len('BQ')+1 : log_path.find('BN')-1])
hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BQ')-1])

ac = core.MLPActorCritic(obs_dim, act_dim, env.act_ll, env.act_ul, BN, DR, Boost_Q, hidden_sizes)
ac.load_state_dict(pi_state_dict)

print("Model has been loaded.")

def get_action(o, deterministic=False): # default: stocastic action
    return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

o, d, ep_ret, ep_len = env.reset(start_conf, goal_pose), False, 0.0, 0.0
while not d:
    # Take deterministic actions at test time
    a = get_action(o, deterministic=True)
    o, r, d, info = env.step(a)
    ep_ret += r
    ep_len += 1
    print("[TEST] Reward: {:4.3f} S:{:4} | DS:{:4} | RS:{:4} | C:{:4} | TO:{:4}".format(
        r, info['is_suc'], info['is_rot_suc'], info['is_dist_suc'], info['is_col'], info['is_timeout']
    ))
print("\n")
env.path = path_refine_and_interpolation(env.path, N=50, c_joints=[2,4,6])
env.visualize_demo(is_second_path=True)
for conf in env.path:
    print("{:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f} {:4.3f}".format(conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]))

# rospy.sleep(1)
# rospy.spin()
