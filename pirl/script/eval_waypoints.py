#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import os
import time
import rospy
import torch
import numpy as np
from envs.Pirl_interpolator import Interpolator
from algos.sac import core
import rospkg
from utils.interpolation_utils import path_interpolation

from pirl_msgs.srv import fk
from pirl_msgs.srv import ik

from envs.utils import PathVisualizer

rospy.init_node("eval_waypoints_main")
rospy.sleep(5)

# ================ [load sac neural interpolator] ================
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'logs/backup/WRwd2_10_0.05_2.5_0.1_5_10_20_5_dth10_rth10__ep_1000_plr_0.0001_qlr_0.0003_pwd_1e-05_qwd_0.001_gclip_1.0_hids_[512, 256, 128]_BQ_1024_BN_False_DR_False'
log_dir = os.path.join(cur_dir_path, log_path)
pi_state_dict = torch.load(os.path.join(log_dir, 'best_model.pt'))

obs_dim = 35
act_dim = 7
act_ll = np.array(rospy.get_param("/robot/vel_ll"))
act_ul = np.array(rospy.get_param("/robot/vel_ul"))

BN = eval(log_path[ log_path.find('BN')+len('BN')+1 : log_path.find('DR')-1])
DR = eval(log_path[ log_path.find('DR')+len('DR')+1 : ])
Boost_Q = eval(log_path[ log_path.find('BQ')+len('BQ')+1 : log_path.find('BN')-1])
hidden_sizes = eval(log_path[ log_path.find('hids')+len('hids')+1 : log_path.find('BQ')-1])

ac = core.MLPActorCritic(obs_dim, act_dim, act_ll, act_ul, BN, DR, Boost_Q, hidden_sizes)
ac.load_state_dict(pi_state_dict)

print("Model has been loaded.")

path_viz = PathVisualizer(is_test=True, is_second_path=True)

# ===================== setup waypoints ==============================
def load_target_ee_poses(path_name, start_pose):
    target_pose_list = [start_pose]
    rospack = rospkg.RosPack()
    file_path = rospack.get_path('pirl') + '/launch/cfg/target_path/' + path_name
    with open(file_path, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            EEpose = []
            P = line.strip().split(';')[0].split(',')
            R = line.strip().split(';')[1].split(',')
            EEpose.append(start_pose[0]+float(P[0]))
            EEpose.append(start_pose[1]+float(P[1]))
            EEpose.append(start_pose[2]+float(P[2]))
            EEpose.append(float(R[1]))
            EEpose.append(float(R[2]))
            EEpose.append(float(R[3]))
            EEpose.append(float(R[0]))
            target_pose_list.append(EEpose)
    return target_pose_list

# base: [-0.086875; 0; 0.37743]
# s_init_waypoints = [[0.8, 0.45, 0.25, 0.0, 0.0, 0.0, 1.0],
#                   [0.8, 0.45, 0.5, 0.0, 0.0, 0.0, 1.0],
#                   [0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
#                   [0.8, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0],
#                   [0.8, -0.45, 0.25, 0.0, 0.0, 0.0, 1.0],
#                   [0.8, -0.45, 0.5, 0.0, 0.0, 0.0, 1.0]]
# interpoled_waypoints = hello_init_waypoints[::5]
# interpoled_waypoints = path_interpolation(np.array(s_init_waypoints) ,50)


# path_name = 's'
# if path_name == 's':
#     start_conf = [0.32859072646481297, 0.180483432860646, 1.2922894312536055, 1.4023895537350413, -0.076182939888603585, -1.7679369578790529, -1.2634014392835944]
#     start_pose = [1.0, 0.3, 0.55, 0.0, 0.0, 0.0, 1.0]
#     interpoled_waypoints = load_target_ee_poses(path_name, start_pose)[::5]
# elif path_name == 'hello_buf':
#     start_conf = [0.32859072646481297, 0.180483432860646, 1.2922894312536055, 1.4023895537350413, -0.076182939888603585, -1.7679369578790529, -1.2634014392835944]
#     start_pose = [0.8, 0.45, 0.25, 0.0, 0.0, 0.0, 1.0]
#     interpoled_waypoints = load_target_ee_poses(path_name, start_pose)[::5]
# elif path_name == 'rot':
#     start_conf = [0.75435628406940525, 0.5875776197718009, 4.5653725089648711, 2.1997877289722894, 3.4548940938617987, 1.3296495814056504, -1.1181117201164357]
#     start_pose = [0.8, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0]
#     interpoled_waypoints = load_target_ee_poses(path_name, start_pose)[::5]
# elif path_name == 'square.txt':
#     start_conf = [0.0272933, -0.0157441, 0.075843, -0.858345, 3.1045, -0.872314, 3.0895853]
#     start_pose = [1.1, 0.0, 0.66, 0.0, 0.0, 0.0, 1.0]
#     interpoled_waypoints = load_target_ee_poses(path_name, start_pose)[::10]

path_name = rospy.get_param("/robot/target_path")
start_conf = rospy.get_param("/robot/init_config")
start_pose = rospy.get_param("/robot/init_pose")
interpoled_waypoints = load_target_ee_poses(path_name, start_pose)[::10]

print("[INFO] # of waypoints: {}".format(len(interpoled_waypoints)))
for i in range(100):
    ipol = Interpolator(ac,start_conf,interpoled_waypoints)
    st = time.time()
    rl_interpolated_conf_list = ipol.execute(deterministic=True)
    print("[INTERPOLATE] elapsed_time: {}".format(time.time()-st))
    rl_ee_path = ipol.fk_output()

    path_viz.pub_arrow(interpoled_waypoints[0], 'b')
    path_viz.pub_arrow(interpoled_waypoints[-1], 'r')

    path_viz.pub_eepath_arrow(interpoled_waypoints)
    path_viz.pub_eepath_strip(interpoled_waypoints)

    path_viz.pub_eepath_arrow(rl_ee_path, second_path=True)
    path_viz.pub_eepath_strip(rl_ee_path, second_path=True)
    path_viz.pub_path(rl_interpolated_conf_list)
    # rospy.sleep(5)

