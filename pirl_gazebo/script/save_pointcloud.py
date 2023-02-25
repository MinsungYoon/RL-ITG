import sys
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time

FOR_TRAIN = eval(sys.argv[1])

rospy.init_node("save_point_cloud")

if FOR_TRAIN:
    scene_number = int(sys.argv[2])
    save_name_cloud = '/data/pirl_data/eval_show/scene_with_box_cloud/pc_'+str(scene_number)+'.npy'
    save_name_occ = '/data/pirl_data/eval_show/scene_with_box_occ/occ_'+str(scene_number)+'.npy'
else:
    save_name_cloud = '/data/pirl_data/eval/scene_cloud/'+ sys.argv[2] +'.npy'
    save_name_occ = '/data/pirl_data/eval/scene_occ/'+ sys.argv[2] +'.npy'

time.sleep(0.2)

def getGrids(points):
    ws_size = [1.0, 1.4, 1.0]
    ws_low = [0.2, -0.7, 0.2]
    res = 0.02

    grid = np.zeros((int(ws_size[0]/res), int(ws_size[1]/res), int(ws_size[2]/res)))

    for p in points:
        grid[int(((p[0] - ws_low[0]) / res))][int(((p[1] - ws_low[1]) / res))][int(((p[2] - ws_low[2]) / res))] = 1.0
    print("[Grid] save.:" + save_name_occ)
    np.save(save_name_occ, grid)

def getPoints(data):
    points = []
    for d in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")):
        points.append(d)
    getGrids(points)
    print("[Point] save.:" + save_name_cloud)
    np.save(save_name_cloud, np.array(points))

rospy.Subscriber("/gridmap3d/occupied_cells", PointCloud2, getPoints)
rospy.sleep(0.4)


