#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import rospy
import time
import numpy as np
from numpy import random as RD

from pirl_msgs.srv import collision
from pirl_msgs.srv import fk
from pirl_msgs.srv import ik
from pirl_msgs.srv import rndsg

from envs.utils import PathVisualizer

rospy.init_node("start_goal_sample_test")
rospy.sleep(1)
rospy.logwarn("[===INFO===] =======> start_goal_sample_test")

rndsg_srv = rospy.ServiceProxy('/rndsg_setting', rndsg)

path_viz = PathVisualizer()

sample_range = [0.2, 1.0, -0.4, 0.4, 0.3, 0.8, -np.pi/3, np.pi/3, 0.2, np.pi/6]
# sample_range = [0.2, 1.0, -0.7, 0.7, 0.3, 1.1, 0.3, np.pi/3]

start_conf_list = []
goal_pos_list = []
while len(start_conf_list)!=1000:
    res = rndsg_srv(sample_range)
    while not res.result:
        res = rndsg_srv(sample_range)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    start_conf_list.append(list(res.start_conf))
    goal_pos_list.append(list(res.goal_pos))


path_viz.pub_path(start_conf_list)
path_viz.pub_eepath_arrow(goal_pos_list)

rospy.logwarn("[===INFO===] <======= start_goal_sample_test")