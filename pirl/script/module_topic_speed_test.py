#! /home/minsungyoon/anaconda3/envs/pirl/bin/python
import rospy
import time
import numpy as np

from pirl_msgs.srv import collision
from pirl_msgs.srv import fk
from pirl_msgs.srv import ik
from pirl_msgs.srv import rndsg


rospy.init_node("topic_test")
rospy.sleep(1)
rospy.logwarn("[===INFO===] =======> topic_test")


cc_srv = rospy.ServiceProxy('/collision_check', collision)
res = cc_srv([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
rospy.logwarn("========col return========")
rospy.logwarn(res.collision_result)
rospy.logwarn(res.result)
log = []
for i in range(100):
    st = time.time()
    res = cc_srv([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    elp_t = time.time()-st
    log.append(elp_t)
    if res.result:
        continue
    else:
        break
print("col: {} times --> avg sec: {}".format(log.__len__(), sum(log)/log.__len__()))

fk_srv = rospy.ServiceProxy('/fk_solver', fk)
res = fk_srv([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
rospy.logwarn("========col return========")
rospy.logwarn(res.fk_result)
rospy.logwarn(res.result)
log = []
for i in range(100):
    st = time.time()
    res = fk_srv([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    elp_t = time.time()-st
    log.append(elp_t)
    if res.result:
        continue
    else:
        break
print("fk: {} times --> avg sec: {}".format(log.__len__(), sum(log)/log.__len__()))



ik_srv = rospy.ServiceProxy('/ik_solver', ik)
res = ik_srv([1.0, 0.3, 0.3, 0.0, 0.0, 0.0, 1.0])
rospy.logwarn("========col return========")
rospy.logwarn(res.ik_result)
rospy.logwarn(res.result)
log = []
for i in range(100):
    st = time.time()
    res = ik_srv([1.0, 0.3, 0.3, 0.0, 0.0, 0.0, 1.0])
    elp_t = time.time()-st
    log.append(elp_t)
    if res.result:
        continue
    else:
        break
print("ik: {} times --> avg sec: {}".format(log.__len__(), sum(log)/log.__len__()))


# col:1.3 msec <- 0.3 msec (C++)
# fk: 1.1 msec <- 0.003 msec (C++)
# ik: 1.8 msec <- 0.6 msec (C++)



rndsg_srv = rospy.ServiceProxy('/rndsg_setting', rndsg)
res = rndsg_srv()
rospy.logwarn("========rndsg return========")
rospy.logwarn(list(res.start_conf)) # type: tuple
rospy.logwarn(np.array(res.goal_pos))
rospy.logwarn(res.result)
log = []
for i in range(100):
    st = time.time()
    res = rndsg_srv()
    elp_t = time.time()-st
    log.append(elp_t)
    if res.result:
        continue
    else:
        break
print("rndsg: {} times --> avg sec: {}".format(log.__len__(), sum(log)/log.__len__()))