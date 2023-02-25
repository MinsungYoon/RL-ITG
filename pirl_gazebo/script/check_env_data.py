#! /root/anaconda2/envs/env/bin/ python

import sys
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

if __name__ == "__main__":
    rospy.init_node("visualize_grid")
    # scene_number = int(sys.argv[1])

    FOR_TRAIN = eval(sys.argv[1])

    if FOR_TRAIN:
        for n_s in range(50):
            grid = np.load('/data/pirl_data/eval_show/scene_with_box_occ/occ_'+str(n_s)+'.npy') # (50, 70, 50)

            ws_size = [1.0, 1.4, 1.0]
            ws_low = [0.2, -0.7, 0.2]   # X: 0.2 ~ 1.2 (m), Y: -0.7 ~ 0.7 (m), Z: 0.2 ~ 1.2 (m)
            res = 0.02                  # Resolution: 0.02 (m)

            publisher = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)
            ma = MarkerArray()

            for i in range(int(ws_size[0]/res)):
                for j in range(int(ws_size[1]/res)):
                    for k in range(int(ws_size[2]/res)):
                        if(grid[i][j][k] == 1):
                            marker = Marker()
                            marker.header.frame_id = "/base_footprint"
                            marker.header.stamp = rospy.Time.now()
                            marker.type = marker.CUBE
                            marker.action = marker.ADD
                            marker.scale.x = res
                            marker.scale.y = res
                            marker.scale.z = res
                            marker.color.a = 1.0
                            marker.color.r = float(i/int(ws_size[0]/res)*0.5 + 0.5)
                            marker.color.g = float(j/int(ws_size[1]/res)*0.5 + 0.5)
                            marker.color.b = float(k/int(ws_size[2]/res)*0.5 + 0.5)
                            marker.pose.orientation.w = 1.0
                            marker.pose.position.x = ws_low[0] + res * i + res/2
                            marker.pose.position.y = ws_low[1] + res * j + res/2
                            marker.pose.position.z = ws_low[2] + res * k + res/2

                            ma.markers.append(marker)

            # Renumber the marker IDs
            id = 0
            for m in ma.markers:
                m.id = id
                id += 1

            # Publish the MarkerArray
            for i in range(5):
                publisher.publish(ma)
                rospy.sleep(0.1)
                # print(i)
            print("=======================> Publish {}-th map".format(n_s))
            for m in ma.markers:
                m.action = Marker.DELETEALL
            publisher.publish(ma)
            print("<======================= delete map")
    else:
        for n_s in ['square']:
            grid = np.load('/data/pirl_data/eval/scene_occ/{}.npy'.format(n_s)) # (50, 70, 50)

            ws_size = [1.0, 1.4, 1.0]
            ws_low = [0.2, -0.7, 0.2]
            res = 0.02

            publisher = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)
            ma = MarkerArray()

            for i in range(int(ws_size[0]/res)):
                for j in range(int(ws_size[1]/res)):
                    for k in range(int(ws_size[2]/res)):
                        if(grid[i][j][k] == 1):
                            marker = Marker()
                            marker.header.frame_id = "/base_footprint"
                            marker.header.stamp = rospy.Time.now()
                            marker.type = marker.CUBE
                            marker.action = marker.ADD
                            marker.scale.x = res
                            marker.scale.y = res
                            marker.scale.z = res
                            marker.color.a = 1.0
                            marker.color.r = 1.0
                            marker.color.g = 0.0
                            marker.color.b = 0.0
                            marker.pose.orientation.w = 1.0
                            marker.pose.position.x = ws_low[0] + res * i + res/2
                            marker.pose.position.y = ws_low[1] + res * j + res/2
                            marker.pose.position.z = ws_low[2] + res * k + res/2

                            ma.markers.append(marker)

            # Renumber the marker IDs
            id = 0
            for m in ma.markers:
                m.id = id
                id += 1

            # Publish the MarkerArray
            for i in range(5):
                publisher.publish(ma)
                rospy.sleep(0.1)
                # print(i)
            print("=======================> Publish {}-th map".format(n_s))
            for m in ma.markers:
                m.action = Marker.DELETEALL
            publisher.publish(ma)
            print("<======================= delete map")
