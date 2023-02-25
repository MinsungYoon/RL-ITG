#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import rospy
import time
import numpy as np
from numpy import random as RD

from pirl_msgs.srv import rndvalconf

from envs.utils import PathVisualizer

rospy.init_node("start_goal_sample_test")
rospy.sleep(1)
rospy.logwarn("[===INFO===] =======> start_goal_sample_test")

rndvalconf_srv = rospy.ServiceProxy('/rndvalconf', rndvalconf)

path_viz = PathVisualizer()

# sample_range = [0.8, 0.8,
#                 0.0, 0.0,
#                 0.6, 0.6,
#                 -np.pi, np.pi,
#                 -np.pi/2, np.pi/2,
#                 -np.pi/2, np.pi/2,
#                 ]

# [WARN!] coordinate: torso_lift_link [-0.086875; 0; 0.37743
sample_range = [0.3+0.086875, 1.1+0.086875,
                -0.4-0, 0.4-0,
                0.3-0.37743, 1.2-0.37743,
                -np.pi, np.pi,
                -np.pi/2, np.pi/2,
                -np.pi/3, np.pi/3
                ]

conf_list = []
pos_list = []
while len(conf_list)!=500:
    res = rndvalconf_srv(sample_range)
    while not res.result:
        res = rndvalconf_srv(sample_range)
    conf_list.append(list(res.configuration))
    pos_list.append(list(res.ee_pose))


path_viz.pub_path(conf_list)
path_viz.pub_eepath_arrow(pos_list)

rospy.logwarn("[===INFO===] <======= start_goal_sample_test")