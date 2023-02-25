#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import rospy
import time
import numpy as np
from numpy import random as RD

from pirl_msgs.srv import collision
from pirl_msgs.srv import scene_set
from pirl_msgs.srv import scene_reset

from pirl_msgs.srv import fk
from pirl_msgs.srv import ik
from pirl_msgs.srv import rndsg

from envs.utils import PathVisualizer

rospy.init_node("col_sample_test")
rospy.sleep(1)
rospy.logwarn("[===INFO===] =======> col_sample_test")

cc_srv = rospy.ServiceProxy('/collision_check', collision)
scene_set_srv = rospy.ServiceProxy('/scene_set', scene_set)
scene_reset_srv = rospy.ServiceProxy('/scene_reset', scene_reset)

path_viz = PathVisualizer()

ul = rospy.get_param('/robot/ul')
ll = rospy.get_param('/robot/ll')

test_count = 0
col_conf_list = []
while len(col_conf_list)!=100:
    rnd_conf = RD.uniform(ll, ul)
    res = cc_srv(rnd_conf.tolist())
    while not res.result:
        res = cc_srv(rnd_conf.tolist())
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    if res.collision_result:
        col_conf_list.append(rnd_conf)
        print("=============================={}".format(len(col_conf_list)))
    test_count +=1
    print(test_count)

path_viz.pub_path(col_conf_list)
rospy.logwarn("[===INFO===] <======= col_sample_test")

for i in range(100):
    res2 = scene_reset_srv()
    if res2.result:
        print("Reset Scene")
    time.sleep(0.1)
    res3 = scene_set_srv(i)
    if res3.result:
        print("Scene #{}".format(i))
    time.sleep(0.1)

import os

data = []
DIR = '/data/torm_data/obs/torm_solution'
k = 0

DIR_LIST = os.listdir(DIR)
DIR_LIST.sort(key=int)
DIR_LIST = DIR_LIST[:500]
for dir_name in DIR_LIST:
    SUB_DIR = os.path.join(DIR, dir_name)
    n_data = 0
    if os.path.isdir(SUB_DIR):
        for f_name in os.listdir(SUB_DIR):
            FILE = os.path.join(SUB_DIR, f_name)
            if os.path.isfile(FILE) and f_name.find("_config.csv") != -1:
                data.append(FILE[:FILE.find("_config.csv")])
                n_data += 1
        print("========== {} ===========: {}".format(dir_name, n_data))
print data
print len(data)

import ipdb
ipdb.set_trace()


# print [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and name.find("_config")!= -1]
# print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and name.find("_config") != -1])
