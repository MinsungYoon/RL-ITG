import os
import csv
import numpy as np
import quaternion  # (w, x, y, z)
import time

c_joints = [2, 4, 6]
def refine_continuous_joints(conf):
    for j in c_joints:
        if conf[j] > np.pi:
            buf = conf[j] - int(conf[j] / (2 * np.pi)) * 2 * np.pi
            if buf > np.pi:
                buf -= 2 * np.pi
            conf[j] = buf
        elif conf[j] < -np.pi:
            buf = conf[j] - int(conf[j] / (2 * np.pi)) * 2 * np.pi
            if buf < -np.pi:
                buf += 2 * np.pi
            conf[j] = buf
    return conf

def calc_pos_and_rot(source):
    res = [source[0], source[1], source[2]]
    eq = np.quaternion(source[6], source[3], source[4], source[5])
    eM = quaternion.as_rotation_matrix(eq)
    res.append(eM[0, 0])
    res.append(eM[1, 0])
    res.append(eM[2, 0])
    res.append(eM[0, 1])
    res.append(eM[1, 1])
    res.append(eM[2, 1])
    return res

def calc_pos_and_rot_error(target, source):
    res = [target[0] - source[0], target[1] - source[1], target[2] - source[2]]
    tq = np.quaternion(target[6], target[3], target[4], target[5])
    tM = quaternion.as_rotation_matrix(tq)
    eq = np.quaternion(source[6], source[3], source[4], source[5])
    eM = quaternion.as_rotation_matrix(eq)
    R_ct = np.matmul(tM, np.linalg.inv(eM)) # rot cur to target
    res.append(R_ct[0, 0])
    res.append(R_ct[1, 0])
    res.append(R_ct[2, 0])
    res.append(R_ct[0, 1])
    res.append(R_ct[1, 1])
    res.append(R_ct[2, 1])
    return res

import rospy
from pirl_msgs.srv import allfk

rospy.init_node("data_load_test")
allfk_srv = rospy.ServiceProxy('/allLinkfk_solver', allfk)

BASE_DIR = "/home/minsungyoon/nvme/torm_data/suc/"
n_train_waypoints = [4, 5, 6, 7]
n_get_train_data_start = [0, 0, 0, 0]
n_get_train_data_end = [900, 900, 900, 400]

n_test_waypoints = [4, 5, 6, 7]
n_get_test_data_start = [900, 900, 900, 400]
n_get_test_data_end = [1000, 1000, 1000, 500]


def pirl_gen_dataset(mode, n_waypoints, n_get_data_start, n_get_data_end):
    DATA_X = []
    DATA_Y = []
    for i_w, w in enumerate(n_waypoints):
        if n_get_data_end[i_w] - n_get_data_start[i_w] > 0:
            data_dir = BASE_DIR + str(w)
            file_configs = [os.path.join(data_dir, "{}_config.csv".format(i))
                            for i in range(n_get_data_start[i_w], n_get_data_end[i_w])]
            file_targetposes = [os.path.join(data_dir, "{}_targetpose.csv".format(i))
                                for i in range(n_get_data_start[i_w], n_get_data_end[i_w])]

            for reverse in range(2):
                for fn_c, fn_t in zip(file_configs, file_targetposes):
                    print("[R{}] In processing... {}".format(reverse, fn_c))
                    configs = []
                    target_poses = []
                    with open(fn_c, 'r') as f:
                        rdr = csv.reader(f)
                        for line in rdr:
                            configs.append(refine_continuous_joints([eval(ele) for ele in line]))
                    with open(fn_t, 'r') as f:
                        rdr = csv.reader(f)
                        for line in rdr:
                            target_poses.append([eval(ele) for ele in line])

                    # processing input and output pair for training.
                    poses = []
                    for conf in configs:
                        poses.append(list(allfk_srv(conf).allfk_result))

                    if reverse == 1:
                        configs.reverse()
                        target_poses.reverse()
                        poses.reverse()

                    n_nodes = len(configs)
                    t = 6
                    gaplist = [1, 2, 3]
                    for gap in gaplist:
                        for i in range(n_nodes):
                            Xi = []
                            Yi = []
                            # [State] total observation: R^133
                            # [State] (1) configuration R: 7
                            Xi += configs[i]

                            # [State] (2) current link position(3), rotation(6) R: (3+6)*8 = 72
                            n_seg = int(len(poses[i]) / 7)
                            for s in range(n_seg):
                                Xi += calc_pos_and_rot(poses[i][7*s+0: 7*s+7])


                            # [Goal] (4) local future target positions relative to current EE frame (t=6) R: 6*(3+6) = 54
                            for f in range(i + gap, i + t * gap + 1, gap):
                                f_idx = f
                                if f_idx >= n_nodes:
                                    f_idx = n_nodes - 1
                                target_pose = target_poses[f_idx]
                                Xi += calc_pos_and_rot_error(target_pose, poses[i][7*(n_seg-1)+0: 7*(n_seg-1)+7])


                            # [Action] delta configs for matching future target poses R: 7
                            target_i = i + gap
                            if target_i >= n_nodes:
                                target_i = n_nodes - 1
                            n_dof = len(configs[i])
                            for k in range(n_dof):
                                diff_j = configs[target_i][k] - configs[i][k]
                                if k in c_joints:
                                    if diff_j > np.pi:
                                        diff_j -= 2 * np.pi
                                    elif diff_j < -np.pi:
                                        diff_j += 2 * np.pi
                                Yi.append(diff_j)

                            DATA_X.append(Xi)
                            DATA_Y.append(Yi)

            # for data statistics
            n_total_costs = []
            n_total_times = []
            n_total_points = []
            n_total_lens = []
            file_cost_logs = [os.path.join(data_dir, "{}_cost_log.csv".format(i))
                              for i in range(n_get_data_start[i_w], n_get_data_end[i_w])]
            file_time_logs = [os.path.join(data_dir, "{}_time_log.csv".format(i))
                              for i in range(n_get_data_start[i_w], n_get_data_end[i_w])]
            file_targetpose_infos = [os.path.join(data_dir, "{}_targetpose_info.csv".format(i))
                                     for i in range(n_get_data_start[i_w], n_get_data_end[i_w])]
            for (fn_c, fn_t, fn_ti) in zip(file_cost_logs, file_time_logs, file_targetpose_infos):
                with open(fn_c, 'r') as f:
                    rdr = csv.reader(f)
                    n_total_costs.append(eval(next(rdr)[-1]))
                with open(fn_t, 'r') as f:
                    rdr = csv.reader(f)
                    n_total_times.append(eval(next(rdr)[-1]))
                with open(fn_ti, 'r') as f:
                    rdr = csv.reader(f)
                    line = next(rdr)
                    n_total_points.append(eval(line[-2]))
                    n_total_lens.append(eval(line[-1]))
            print(
            "[INFO] Mode: {}, N_wyp: {:4.3f}| Cost: {:4.3f}({:4.3f})| Time: {:4.3f}({:4.3f})| TargetPoint: {:4.3f}({:4.3f})| TargetLen: {:4.3f}({:4.3f})".format(
                mode, w,
                np.array(n_total_costs).mean(), np.array(n_total_costs).std(),
                np.array(n_total_times).mean(), np.array(n_total_times).std(),
                np.array(n_total_points).mean(), np.array(n_total_points).std(),
                np.array(n_total_lens).mean(), np.array(n_total_lens).std(),
            ))
            print("[INFO] Total Points: {:4.3f}| Total Len: {:4.3f}".format(
                np.array(n_total_points).sum(), np.array(n_total_lens).sum()
            ))
            print("[INFO] Total DATA: {:4.3f}".format(
                len(DATA_X)
            ))

            np_DATA_X = np.array(DATA_X, dtype=np.float64)
            np_DATA_Y = np.array(DATA_Y, dtype=np.float64)

            np.savez("./{}_wps_data_{}.npz".format(w, mode), x=np_DATA_X, y=np_DATA_Y)

            DATA_X = []
            DATA_Y = []


pirl_gen_dataset('train', n_train_waypoints, n_get_train_data_start, n_get_train_data_end)
pirl_gen_dataset('test', n_test_waypoints, n_get_test_data_start, n_get_test_data_end)
