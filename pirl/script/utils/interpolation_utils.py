import numpy as np
import torch
import torch.nn as nn

# convert continuous joints to range from -pi to pi.
def refine_continuous_joints(conf_list, c_joints=None):
    for conf in conf_list:
        if c_joints:
            for j in c_joints:
                if conf[j] > np.pi:
                    buf = conf[j] - int(conf[j]/(2*np.pi))*2*np.pi
                    if buf > np.pi:
                        buf -= 2*np.pi
                    conf[j] = buf
                elif conf[j] < -np.pi:
                    buf = conf[j] + int(conf[j]/(2*np.pi))*2*np.pi
                    if buf < -np.pi:
                        buf += 2*np.pi
                    conf[j] = buf
    return conf_list

def path_interpolation(conf_list, N=100, c_joints=None):
    interpolated_path = []
    N_waypoints = len(conf_list)
    N_interval = N_waypoints - 1
    n_inter_points = [N // N_interval] * N_interval
    for i in range(N % N_interval):
        n_inter_points[i] += 1

    for i in range(N_interval):
        df = conf_list[i + 1] - conf_list[i]
        c_check_joint_list = []
        if c_joints:
            for j_idx in c_joints:
                if df[j_idx] > np.pi:
                    df[j_idx] = df[j_idx] - 2*np.pi
                    c_check_joint_list.append(j_idx)
                elif df[j_idx] < -np.pi:
                    df[j_idx] = 2*np.pi + df[j_idx]
                    c_check_joint_list.append(j_idx)
        for k in range(n_inter_points[i]):
            buf = conf_list[i] + k * df / n_inter_points[i]
            if c_check_joint_list:
                for j_idx in c_check_joint_list:
                    if buf[j_idx] < -np.pi and conf_list[i][j_idx] < 0:
                        buf[j_idx] += 2*np.pi
                    elif buf[j_idx] > np.pi and conf_list[i][j_idx] > 0:
                        buf[j_idx] -= 2*np.pi
            interpolated_path.append(buf)
    interpolated_path.append(conf_list[-1])
    return interpolated_path

def path_refine_and_interpolation(conf_list, N=100, c_joints=None):
    conf_list = refine_continuous_joints(conf_list, c_joints)
    return path_interpolation(conf_list, N, c_joints)

