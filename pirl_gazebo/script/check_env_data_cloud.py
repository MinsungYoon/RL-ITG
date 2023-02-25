#! /root/anaconda2/envs/env/bin/ python

import sys
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud

if __name__ == "__main__":
    rospy.init_node("visualize_grid")
    # scene_number = int(sys.argv[1])

    FOR_TRAIN = eval(sys.argv[1])

    for n_s in range(100):
        c_data = np.load('/data/pirl_data/eval_show/scene_with_box_cloud/pc_'+str(n_s)+'.npy') # (50, 70, 50)

        ws_size = [1.0, 1.4, 1.0]
        ws_low = [0.2, -0.7, 0.2]   # X: 0.2 ~ 1.2 (m), Y: -0.7 ~ 0.7 (m), Z: 0.2 ~ 1.2 (m)
        res = 0.02                  # Resolution: 0.02 (m)

        publisher = rospy.Publisher('/cloud_data', PointCloud, queue_size=10)
        PC = PointCloud()
        PC.header.frame_id = "/base_link"
        # PC.points = c_data.shape[0]
        # PC.channels = c_data.shape[1]

        for i in range(c_data.shape[0]):
            point = Point()
            point.x = c_data[i][0]
            point.y = c_data[i][1]
            point.z = c_data[i][2]
            PC.points.append(point)

        # Publish the MarkerArray
        for i in range(1):
            publisher.publish(PC)
            rospy.sleep(0.03)
            # print(i)
        print("=======================> Publish {}-th map".format(n_s))
