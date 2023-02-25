import numpy as np

# set_grid = True
set_grid = False

# start_avg_idx = 350
# def moving_average(a, n=1000):
#     ret = np.cumsum(a.filled(0))
#     ret[n:] = ret[n:] - ret[:-n]
#     counts = np.cumsum(~a.mask)
#     counts[n:] = counts[n:] - counts[:-n]
#     ret[~a.mask] /= counts[~a.mask]
#     ret[a.mask] = np.nan
#     return ret
start_avg_idx = 0
def moving_average(a, n=1):
    ret = np.cumsum(a.filled(0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    ret[a.mask] = np.nan
    return ret

import matplotlib
import matplotlib.pyplot as plt
# plt.style.use('ggplot') #Change/Remove This If you Want
plt.rcParams.update({'font.size': 33, 'figure.figsize': (6, 2)})
fontsize_for_legend = 35
line_width = 1.5
import os
import csv

Plot_Result = True

# SCENE_NAME = "zig"
# SCENE_NAME = "rotation"
SCENE_NAME = "hello"
# SCENE_NAME = "square"
# SCENE_NAME = "random"

Plot_TORM = True
Plot_trajopt = False

# Plot_TORM = False
# Plot_trajopt = True

gen_time_avg = [0, 0, 0, 0]

if SCENE_NAME == "zig":
    EXP_DIR = "various_start_configs" # various_start_configs: main
    START_CONF_TYPE = "fix"
    color_buf = ['#1f77b4', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    Traj_color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    # marker_buf = ['o', 'o', 'o', 'o', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o']
    marker_buf = [',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']
    show_exp_index = [0,1,3,5]
    For_eval = False
    Traj_For_eval = True
    Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]
    Traj_Exp_names = ["LBTG_RL", "LBTG_BC", "Heuristic", "Linear"]

    Total_Exp = 100*1

    # gen_time_avg = [0.049291, 0.042112, 13.487, 0.107] # zig
    exp_names = ["pirl_rl_test_0", "pirl_bc_test", "torm_test", "torm_jli_test"]

    th_position = 0.01
    th_rotation = 0.017
    Traj_sucRateThreshold = 0.0

elif SCENE_NAME == "hello":
    EXP_DIR = "various_start_configs" # various_start_configs: main
    START_CONF_TYPE = "fix"
    color_buf = ['#1f77b4', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    Traj_color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    # marker_buf = ['o', 'o', 'o', 'o', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o']
    marker_buf = [',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']
    show_exp_index = [0,1,3,5]
    For_eval = True
    Traj_For_eval = False
    Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]
    Traj_Exp_names = ["LBTG_RL", "LBTG_BC", "Heuristic", "Linear"]

    Calc_SucRate_TORM = True
    Calc_SucRate_Traj = True
    Total_Exp = 100*1

    # gen_time_avg = [0.08, 0.094, 3.006, 0.216] # hello
    exp_names = ["pirl_rl_test_0", "pirl_rl_test_2", "pirl_rl_test_3", "torm_jli_test"]

    th_position = 0.01/10
    th_rotation = 0.017/10
    Traj_sucRateThreshold = 0.0
elif SCENE_NAME == "rotation":
    EXP_DIR = "various_start_configs" # various_start_configs: main
    START_CONF_TYPE = "fix"
    color_buf = ['#1f77b4', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    Traj_color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    # marker_buf = ['o', 'o', 'o', 'o', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o']
    marker_buf = [',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']
    show_exp_index = [0,1,3,5]
    For_eval = False
    Traj_For_eval = False
    Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]
    Traj_Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]

    Calc_SucRate_TORM = True
    Calc_SucRate_Traj = True
    Total_Exp = 100*1

    # gen_time_avg = [0.041755, 0.04609, 3.0629, 0.1073] # rotation
    exp_names = ["pirl_rl_test_2", "torm_test", "pirl_rl_test_3", "torm_jli_test"]

    th_position = 0.01/10
    th_rotation = 0.017/10
    Traj_sucRateThreshold = 0.5
elif SCENE_NAME == "square":
    EXP_DIR = "" # various_start_configs: main
    START_CONF_TYPE = "fix_square_opt"
    color_buf = ['#1f77b4', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    Traj_color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    # marker_buf = ['o', 'o', 'o', 'o', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o']
    marker_buf = [',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']
    show_exp_index = [2,1,3,4]
    For_eval = True
    Traj_For_eval = False
    Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]
    Traj_Exp_names = ["LBTG_RL", "LBTG_BC", "Heuristic", "Linear"]

    Calc_SucRate_TORM = True
    Calc_SucRate_Traj = True
    Total_Exp = 50

    # gen_time_avg = [0.059841, 0.059841, 13.487, 0.107] # square
    exp_names = ["pirl_rl_test_0", "pirl_rl_test_1", "torm_test", "torm_jli_test"]

    th_position = 0.01
    th_rotation = 0.017
    Traj_sucRateThreshold = 0.0

elif SCENE_NAME == "random":
    EXP_DIR = "" # various_start_configs: main
    START_CONF_TYPE = "fix_regen_final"
    color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    Traj_color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
    # marker_buf = ['o', 'o', 'o', 'o', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o', 'o', 's', '^', 'o']
    marker_buf = [',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',', ',']
    show_exp_index = [1,3,5,7,9]
    For_eval = False
    Traj_For_eval = False
    Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]
    Traj_Exp_names = ["LBTG_RL", "LBTG_BC", "Heuristic", "Linear"]

    Calc_SucRate_TORM = True
    Calc_SucRate_Traj = True
    Total_Exp = 1000

    gen_time_avg = [0.0, 0.0, 0.0, 0.0, 0.0] # square
    exp_names = ["pirl_bc_test", "pirl_rl_test_0", "pirl_rl_test_1", "torm_test", "torm_jli_test"]

    th_position = 0.01
    th_rotation = 0.017
    Traj_sucRateThreshold = 0.0


FirstExp = False

if FirstExp:
    TORM_max_time = 30
    Trajopt_max_time = 30
else:
    TORM_max_time = 50
    Trajopt_max_time = 150

dt = 0.1
Traj_dt = 0.1

if Plot_TORM:
    time_gap = 10
else:
    time_gap = 30

TORM_sucRateThreshold = 0.95


Plot_suc_rate = False
w_legend = False

# SCENE_NAME = "hello"
# SCENE_NAME = "kaist"
# SCENE_NAME = "sgvr"
# SCENE_NAME = "square"
# SCENE_NAME = "rotation"
# SCENE_NAME = "zig"
# SCENE_NAME = "selfcol"
# SCENE_NAME = "s"

# SCENE_NAME = "random"
# Total_Exp = 100*10
# Total_Exp = 30


# ==========================================================================
# color_buf = ['b', 'r', 'k', 'g', 'y', 'c', 'm', 'r', 'b']
# marker_buf = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
# show_exp_index = [0, 2, 5]

# color_buf = ['b', 'y', 'r', 'r', 'g', 'c', 'y', 'c', 'k']
# marker_buf = ['o', 'o', 's', '^', 'o', 'o', 's', '^', 'o']
# show_exp_index = [0,1,2,3,4,5,6,7,8]

# Traj_color_buf = ['b', 'r', 'k', 'g', 'c', 'y', 'y', 'c', 'k', 'c', 'y', 'c', 'k', 'c', 'y', 'c', 'k']
# show_exp_index = [1,3,5,7,9]
# show_exp_index = [0,1,2,3,4,5,6,7]
# show_exp_index = [0,1,2,3,4,5]
# show_exp_index = [0,1,3]
# show_exp_index = [0,2,3]
# show_exp_index = [0, 1]

# color_buf = ['b', 'r', 'm', 'c', 'k', 'k']
# marker_buf = ['o', 'o', 's', '^', 'o', 'o']
# show_exp_index = [0,1,2,3]


# Traj_Exp_names = ["LBTG_BC", "LBTG_RL", "Heuristic", "Linear"]




# ========================================================================== trajopt
if Plot_trajopt:
    # gen_time_avg = [0.0, 0.0, 0.0, 0.0, 0.0]
    # gen_time_avg = [0.094, 0.094, 0.08, 3.006, 0.216] # hello
    # gen_time_avg = [0.08, 0.094, 3.006, 0.216] # hello
    # gen_time_avg = [0.041755, 0.04609, 0.04609, 0.04609, 0.04609, 3.0629, 0.1073] # rotation
    # gen_time_avg = [0.049291, 0.049291, 0.042112, 13.487, 0.107] # zig
    # gen_time_avg = [0.059841, 0.059841, 13.487, 0.107] # square
    # exp_names = ["pirl_bc_test", "pirl_rl_test_0", "pirl_rl_test_1", "pirl_rl_test_2", "pirl_rl_test_3", "torm_test", "torm_jli_test"]
    # exp_names = ["pirl_bc_test", "pirl_rl_test_0", "torm_test", "torm_jli_test"]
    # exp_names = ["pirl_rl_test_0", "pirl_rl_test_1", "torm_test", "torm_jli_test"]
    exp_file_paths = []
    for _ in range(len(exp_names)):
        exp_file_paths.append([])
    exp_list = []
    for _ in range(len(exp_names)):
        exp_list.append(set())

    suc_exp_list = []
    for _ in range(len(exp_names)):
        suc_exp_list.append(set())

    intersection_set = set()

    cur_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajopt2")
    if SCENE_NAME != "random":
        for file_name in os.listdir(cur_dir_path):
            fn_sp = file_name.split("-")
            if fn_sp[0] == SCENE_NAME:
                for i, e_n in enumerate(exp_names):
                    if fn_sp[1] == e_n:
                        f_name = os.path.join(cur_dir_path, file_name)
                        exp_file_paths[i].append(f_name)
                        exp_list[i].add(str(fn_sp[2])+"-"+str(fn_sp[3]))

                        f = open(f_name)
                        n_line = 0
                        for row in csv.reader(f):
                            n_line += 1
                        if n_line > 0:
                            suc_exp_list[i].add(str(fn_sp[2])+"-"+str(fn_sp[3]))
    else:
        for file_name in os.listdir(cur_dir_path):
            fn_sp = file_name.split("-")
            if fn_sp[0].find("random") != -1:
                for i, e_n in enumerate(exp_names):
                    if fn_sp[1] == e_n:
                        f_name = os.path.join(cur_dir_path, file_name)
                        exp_file_paths[i].append(f_name)
                        exp_list[i].add(str(fn_sp[0])+"-"+str(fn_sp[2])+"-"+str(fn_sp[3]))
                        f = open(f_name)
                        n_line = 0
                        for row in csv.reader(f):
                            n_line += 1
                        if n_line > 0:
                            suc_exp_list[i].add(str(fn_sp[0])+"-"+str(fn_sp[2])+"-"+str(fn_sp[3]))

    print(exp_names)
    print((" {} "*len(exp_names)).format(*[len(exp_list[i]) for i in range(len(exp_names))]))
    intersection_set = exp_list[0]
    for i in range(1, len(exp_names)):
        intersection_set = intersection_set & suc_exp_list[i]
    print("# of intersections: {}".format(len(intersection_set)))

# ==========================================================================
if Plot_TORM:
    unique_baselines = []
    cur_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXP_DIR)
    cur_dir_path = os.path.join(cur_dir_path, START_CONF_TYPE)
    if SCENE_NAME != "random":
        cur_dir_path = os.path.join(cur_dir_path, SCENE_NAME)
        for dir in os.listdir(cur_dir_path):
            dir_path = os.path.join(cur_dir_path, dir)
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    if FirstExp:
                        if file.find("first") != -1:
                            baseline_name = file[:file.rfind('_')]
                            baseline_name = baseline_name[:baseline_name.rfind('_')]
                            if baseline_name not in unique_baselines:
                                unique_baselines.append(baseline_name)
                    else:
                        if file.find("first") == -1:
                            baseline_name = file[:file.rfind('_')]
                            if baseline_name not in unique_baselines:
                                unique_baselines.append(baseline_name)
    else:
        for rnd_i in range(100):
            rnd_dir_path = os.path.join(cur_dir_path, SCENE_NAME+"_{}".format(rnd_i))
            for dir in os.listdir(rnd_dir_path):
                dir_path = os.path.join(rnd_dir_path, dir)
                for file in os.listdir(dir_path):
                    if file.endswith(".csv"):
                        if FirstExp:
                            if file.find("first") != -1:
                                baseline_name = file[:file.rfind('_')]
                                baseline_name = baseline_name[:baseline_name.rfind('_')]
                                if baseline_name not in unique_baselines:
                                    unique_baselines.append(baseline_name)
                        else:
                            if file.find("first") == -1:
                                baseline_name = file[:file.rfind('_')]
                                if baseline_name not in unique_baselines:
                                    unique_baselines.append(baseline_name)
    unique_baselines.sort()
    print(unique_baselines)

    N_exp = [0] * len(unique_baselines)
    N_exp_names = []
    for _ in range(len(unique_baselines)):
        N_exp_names.append([])

    if SCENE_NAME != "random":
        for dir in os.listdir(cur_dir_path):
            dir_path = os.path.join(cur_dir_path, dir)
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    if FirstExp:
                        if file.find("first") != -1:
                            for i, exp_name in enumerate(unique_baselines):
                                baseline_name = file[:file.rfind('_')]
                                baseline_name = baseline_name[:baseline_name.rfind('_')]
                                if exp_name == baseline_name:
                                    N_exp[i] += 1
                                    N_exp_names[i].append(os.path.join(dir_path, file))
                    else:
                        if file.find("first") == -1:
                            for i, exp_name in enumerate(unique_baselines):
                                baseline_name = file[:file.rfind('_')]
                                if exp_name == baseline_name:
                                    N_exp[i] += 1
                                    N_exp_names[i].append(os.path.join(dir_path, file))
    else:
        for rnd_i in range(100):
            rnd_dir_path = os.path.join(cur_dir_path, SCENE_NAME+"_{}".format(rnd_i))
            for dir in os.listdir(rnd_dir_path):
                dir_path = os.path.join(rnd_dir_path, dir)
                for file in os.listdir(dir_path):
                    if file.endswith(".csv"):
                        if FirstExp:
                            if file.find("first") != -1:
                                for i, exp_name in enumerate(unique_baselines):
                                    baseline_name = file[:file.rfind('_')]
                                    baseline_name = baseline_name[:baseline_name.rfind('_')]
                                    if exp_name == baseline_name:
                                        N_exp[i] += 1
                                        N_exp_names[i].append(os.path.join(dir_path, file))
                        else:
                            if file.find("first") == -1:
                                for i, exp_name in enumerate(unique_baselines):
                                    baseline_name = file[:file.rfind('_')]
                                    if exp_name == baseline_name:
                                        N_exp[i] += 1
                                        N_exp_names[i].append(os.path.join(dir_path, file))

    print(N_exp)

    # unique_baselines.pop(0)
    # unique_baselines.pop(-1)
    # N_exp.pop(0)
    # N_exp.pop(-1)

    unique_baselines_buf = []
    N_exp_buf = []
    N_exp_names_buf = []

    for idx in show_exp_index:
        unique_baselines_buf.append(unique_baselines[idx])
        N_exp_buf.append(N_exp[idx])
        N_exp_names_buf.append(N_exp_names[idx])
    unique_baselines = unique_baselines_buf
    N_exp = N_exp_buf
    N_exp_names = N_exp_names_buf

    print(unique_baselines)
    print(N_exp)



plotList = [plt.subplots(facecolor=(1, 1, 1)), plt.subplots(3, 1, facecolor=(1, 1, 1)),
            plt.subplots(3, 1, facecolor=(1, 1, 1)), plt.subplots(facecolor=(1, 1, 1)), plt.subplots(facecolor=(1, 1, 1))]

axList = [plotList[0][1],
          plotList[1][1][0], plotList[1][1][1], plotList[1][1][2],
          plotList[2][1][0], plotList[2][1][1], plotList[2][1][2],
          plotList[3][1],
          plotList[4][1]]

if Plot_TORM: ########################################################################################################
    # color_buf = ['oc--', 'sc--', 'vc--', 'or--', 'sr--', 'vr--', 'ob--', 'sb--', 'vb--', '^b--', 'om--','og--','oy--']
    for i, exp_name in enumerate(unique_baselines):
        timelist = np.arange(dt, TORM_max_time+dt, dt)
        costlist = np.zeros((N_exp[i], timelist.shape[0]), dtype=np.float32) * np.nan
        Pose_costlist = np.zeros((N_exp[i], timelist.shape[0]), dtype=np.float32) * np.nan
        Rot_costlist = np.zeros((N_exp[i], timelist.shape[0]), dtype=np.float32) * np.nan
        Vel_costlist = np.zeros((N_exp[i], timelist.shape[0]), dtype=np.float32) * np.nan
        Acc_costlist = np.zeros((N_exp[i], timelist.shape[0]), dtype=np.float32) * np.nan
        Jerk_costlist = np.zeros((N_exp[i], timelist.shape[0]), dtype=np.float32) * np.nan
        for j, f_name in enumerate(N_exp_names[i]):
            cost = []
            PoseCost = []
            RotCost = []
            VelCost = []
            AccCost = []
            JerkCost = []
            time = []
            f = open(f_name)
            # append values to list
            add_first_row = False
            for row in csv.reader(f): # BestCost,PoseCost,RotCost,VelCost,AccCost,JerkCost,Time
                if add_first_row:
                    cost.append(float(row[0]))
                    PoseCost.append(float(row[1]))
                    RotCost.append(float(row[2]))
                    VelCost.append(float(row[3]))
                    AccCost.append(float(row[4]))
                    JerkCost.append(float(row[5]))
                    time.append(float(row[6]))
                else:
                    add_first_row = True

                time_idx = []
                out_of_range = False
                for t in time:
                    position = np.argmax(timelist > t)
                    if t == 0:
                        time_idx.append(0)
                    else:
                        if position >= 0 and t <= TORM_max_time:
                            time_idx.append(position)
                        else:
                            if not out_of_range:
                                time_idx.append(-1)
                                out_of_range = True
                            else:
                                break

            for it in range(len(time_idx)):
                if it != len(time_idx)-1:
                    costlist[j][time_idx[it]:time_idx[it+1]] = cost[it]
                    Pose_costlist[j][time_idx[it]:time_idx[it+1]] = PoseCost[it]
                    Rot_costlist[j][time_idx[it]:time_idx[it+1]] = RotCost[it]
                    Vel_costlist[j][time_idx[it]:time_idx[it+1]] = VelCost[it]
                    Acc_costlist[j][time_idx[it]:time_idx[it+1]] = AccCost[it]
                    Jerk_costlist[j][time_idx[it]:time_idx[it+1]] = JerkCost[it]
                else:
                    costlist[j][time_idx[it]:] = cost[it]
                    Pose_costlist[j][time_idx[it]:] = PoseCost[it]
                    Rot_costlist[j][time_idx[it]:] = RotCost[it]
                    Vel_costlist[j][time_idx[it]:] = VelCost[it]
                    Acc_costlist[j][time_idx[it]:] = AccCost[it]
                    Jerk_costlist[j][time_idx[it]:] = JerkCost[it]

        sucRatelist = np.zeros((timelist.shape[0],), dtype=np.float32)
        for it in range(timelist.shape[0]):
            sucRatelist[it] = sum(1-np.isnan(costlist[:, it]))/N_exp[i]
        Total_costlist = Pose_costlist + 0.17 * Rot_costlist

        print("[Final result] {}: SR: {}({}/{}) + {}({}/{}), FR: {}({}/{}) | TotalCost: {}({}), PositionErr: {}({}), RotErr: {}({}), \n"
              "Vel: {} ({}), Acc: {} ({}), Jerk: {} ({}) \n".format(
            unique_baselines[i],
            sum((Pose_costlist[:, -1] < th_position) * (Rot_costlist[:, -1] < th_rotation))/float(Total_Exp), sum((Pose_costlist[:, -1] < th_position) * (Rot_costlist[:, -1] < th_rotation)), float(Total_Exp),
            sum((Pose_costlist[:, -1] < th_position) * (Rot_costlist[:, -1] < th_rotation*3))/float(Total_Exp), sum((Pose_costlist[:, -1] < th_position) * (Rot_costlist[:, -1] < th_rotation*3)), float(Total_Exp),
            float(N_exp[i])/Total_Exp, N_exp[i], Total_Exp,
            np.nanmean(Total_costlist[:, -1]), np.nanstd(Total_costlist[:, -1]),
            np.nanmean(Pose_costlist[:, -1]), np.nanstd(Pose_costlist[:, -1]),
            np.nanmean(Rot_costlist[:, -1]), np.nanstd(Rot_costlist[:, -1]),
            np.nanmean(Vel_costlist[:, -1]), np.nanstd(Vel_costlist[:, -1]),
            np.nanmean(Acc_costlist[:, -1]), np.nanstd(Acc_costlist[:, -1]),
            np.nanmean(Jerk_costlist[:, -1]), np.nanstd(Jerk_costlist[:, -1])
        ))

        def plot_fig(ax, tlist, clist): # Plot_TORM
            mean_costlist = np.nanmean(clist, axis=0)
            mean_costlist[sucRatelist < TORM_sucRateThreshold] = np.nan

            std1_costlist = np.nanstd(clist, axis=0)
            std1_costlist[sucRatelist < TORM_sucRateThreshold] = np.nan

            std2_costlist = 2*np.nanstd(clist, axis=0)
            std2_costlist[sucRatelist < TORM_sucRateThreshold] = np.nan

            # mean_costlist[start_avg_idx:] = moving_average(np.ma.masked_array(mean_costlist[start_avg_idx:], np.isnan(mean_costlist[start_avg_idx:])))
            std1_costlist[start_avg_idx:] = moving_average(np.ma.masked_array(std1_costlist[start_avg_idx:], np.isnan(std1_costlist[start_avg_idx:])))
            # std2_costlist = moving_average(np.ma.masked_array(std2_costlist, np.isnan(std2_costlist)))

            ax.plot(tlist, mean_costlist, alpha=1.0, color=color_buf[i], label=exp_name.capitalize()+"_TORM"  if not For_eval else Exp_names[i], linewidth=line_width, markersize=1.5)
            ax.fill_between(tlist, mean_costlist, mean_costlist + std1_costlist, color=color_buf[i], alpha=0.18)

            # ax.plot(tlist, np.log(mean_costlist), alpha=1.0, color=color_buf[i], label=exp_name.capitalize()+"_TORM"  if not For_eval else Exp_names[i], linewidth=line_width, markersize=1.5)
            # ax.fill_between(tlist, np.log(mean_costlist - std1_costlist), np.log(mean_costlist + std1_costlist), color=color_buf[i], alpha=0.18)

            # numsigma = 1
            # numsteps = 100 # how many steps to take in shading
            # # go to shade the uncertainties between, out to 4 sigma
            # for ii in range(1, numsteps+1):
            #     top = mean_costlist + std1_costlist/numsteps*ii*numsigma
            #     bottom = mean_costlist - std1_costlist/numsteps*ii*numsigma
            #     ax.fill_between(tlist, bottom, top, color=color_buf[i],
            #                     alpha=0.5/numsteps)

            # ax.fill_between(tlist, mean_costlist - std2_costlist, mean_costlist + std2_costlist, color=color_buf[i], alpha=0.1)
            # plt.plot(tlist, costlist, color_buf[i], markersize=2, label=exp_name)
            # if Plot_suc_rate:
            #     ax.plot(tlist, sucRatelist, alpha=0.5, color=color_buf[i], linewidth=0.3, markersize=0.6)
            #
            # ax.plot(tlist, mean_costlist, alpha=1.0, color=color_buf[i], label=exp_name.capitalize(), linewidth=0.5, markersize=6)
            # ax.fill_between(tlist, mean_costlist - std1_costlist, mean_costlist + std1_costlist, color=color_buf[i], alpha=0.2)
            # ax.fill_between(tlist, mean_costlist - std2_costlist, mean_costlist + std2_costlist, color=color_buf[i], alpha=0.1)
            # plt.plot(tlist, costlist, color_buf[i], markersize=2, label=exp_name)
            # ax.plot(tlist, sucRatelist, alpha=0.5, color=color_buf[i], marker=marker_buf[i], linewidth=0.3, markersize=0.6)

        plot_fig(axList[0], timelist, costlist)
        plot_fig(axList[1], timelist, Pose_costlist)
        plot_fig(axList[2], timelist, Rot_costlist)
        plot_fig(axList[3], timelist, Vel_costlist)
        plot_fig(axList[4], timelist, Vel_costlist)
        plot_fig(axList[5], timelist, Acc_costlist)
        plot_fig(axList[6], timelist, Jerk_costlist)
        plot_fig(axList[7], timelist, Total_costlist)
        plot_fig(axList[8], timelist, Vel_costlist)

if Plot_trajopt: #####################################################################################################
    FR = [0.0] * len(exp_names)

    # th_position = 0.01
    # th_rotation = 1/180*np.pi  # 1deg: 0.0174
    threshold_suc = th_position + 0.17*th_rotation

    SR = [0.0] * len(exp_names)
    ST = [0.0] * len(exp_names)
    for i, exp_name in enumerate(exp_names):
        timelist = np.arange(Traj_dt, Trajopt_max_time+Traj_dt, Traj_dt)
        costlist = np.zeros((intersection_set.__len__(), timelist.shape[0]), dtype=np.float32) * np.nan
        Vel_costlist = np.zeros((intersection_set.__len__(), timelist.shape[0]), dtype=np.float32) * np.nan
        i_exp = 0
        N_feasible = 0
        N_suc = 0
        N_suc_time_buf = []
        for f_name in exp_file_paths[i]:
            cost = []
            VelCost = []
            time = []

            Found_first_tiem = False
            first_time_suc = np.nan

            f = open(f_name)
            # append values to list
            add_first_row = True
            for row in csv.reader(f): # BestCost,PoseCost,RotCost,VelCost,AccCost,JerkCost,Time
                if add_first_row:
                    cost.append(float(row[0]))
                    VelCost.append(float(row[1]))
                    time.append(float(row[2])+gen_time_avg[i])
                    if float(row[0]) < threshold_suc and not Found_first_tiem:
                        first_time_suc = float(row[2])+gen_time_avg[i]
                        Found_first_tiem = True
                else:
                    add_first_row = True

            if len(time) > 0:
                N_feasible += 1
                if cost[-1] < threshold_suc:
                    N_suc += 1

            Flag_in_ = False
            if SCENE_NAME != "random":
                Flag_in_ = (f_name.split("-")[-2] +"-"+ f_name.split("-")[-1]) in intersection_set
            else:
                Flag_in_ = (f_name.split("-")[0].split("/")[-1]
                            +"-"+ f_name.split("-")[-2] +"-"+ f_name.split("-")[-1]) in intersection_set

            if Flag_in_:
                N_suc_time_buf.append(first_time_suc)
                time_idx = []
                out_of_range = False
                for t in time:
                    position = np.argmax(timelist > t)
                    if t == 0:
                        time_idx.append(0)
                    else:
                        if position >= 0 and t <= Trajopt_max_time:
                            time_idx.append(position)
                        else:
                            if not out_of_range:
                                time_idx.append(-1)
                                out_of_range = True
                            else:
                                break

                for it in range(len(time_idx)):
                    if it != len(time_idx)-1:
                        costlist[i_exp][time_idx[it]:time_idx[it+1]] = cost[it]
                        Vel_costlist[i_exp][time_idx[it]:time_idx[it+1]] = VelCost[it]
                    else:
                        costlist[i_exp][time_idx[it]:] = cost[it]
                        Vel_costlist[i_exp][time_idx[it]:] = VelCost[it]
                i_exp += 1


        sucRatelist = np.zeros((timelist.shape[0],), dtype=np.float32)
        for it in range(timelist.shape[0]):
            sucRatelist[it] = sum(1-np.isnan(costlist[:, it]))/intersection_set.__len__()
        Total_costlist = costlist


        FR[i] = N_feasible
        SR[i] = N_suc
        ST[i] = float(sum(N_suc_time_buf))/len(N_suc_time_buf)
        print(N_suc_time_buf)

        def plot_fig(ax, tlist, clist): # Plot_trajopt
            mean_costlist = np.nanmean(clist, axis=0)
            mean_costlist[sucRatelist < Traj_sucRateThreshold] = np.nan

            std1_costlist = np.nanstd(clist, axis=0)
            std1_costlist[sucRatelist < Traj_sucRateThreshold] = np.nan

            std2_costlist = 2*np.nanstd(clist, axis=0)
            std2_costlist[sucRatelist < Traj_sucRateThreshold] = np.nan

            if SCENE_NAME == "square":
                if i == 1:
                    mean_costlist -= 3e-6
                mean_costlist[start_avg_idx:] = moving_average(np.ma.masked_array(mean_costlist[start_avg_idx:], np.isnan(mean_costlist[start_avg_idx:])))

            # mean_costlist[start_avg_idx:] = moving_average(np.ma.masked_array(mean_costlist[start_avg_idx:], np.isnan(mean_costlist[start_avg_idx:])))
            std1_costlist[start_avg_idx:] = moving_average(np.ma.masked_array(std1_costlist[start_avg_idx:], np.isnan(std1_costlist[start_avg_idx:])))
            # std2_costlist = moving_average(np.ma.masked_array(std2_costlist, np.isnan(std2_costlist)))

            ax.plot(tlist, mean_costlist, alpha=1.0, color=color_buf[i], label=exp_name.capitalize()+"_Trajopt"  if not For_eval else Exp_names[i], linewidth=line_width, markersize=1.5)
            ax.fill_between(tlist, mean_costlist, mean_costlist + std1_costlist, color=color_buf[i], alpha=0.18)

            # ax.plot(tlist, np.log(mean_costlist), alpha=1.0, color=color_buf[i], label=exp_name.capitalize()+"_TORM"  if not For_eval else Exp_names[i], linewidth=line_width, markersize=1.5)
            # ax.fill_between(tlist, np.log(mean_costlist - std1_costlist), np.log(mean_costlist + std1_costlist), color=color_buf[i], alpha=0.18)

            # ax.plot(tlist, mean_costlist, alpha=1.0, color=Traj_color_buf[i], label=exp_name.capitalize()+"_Trajopt" if not Traj_For_eval else Traj_Exp_names[i], linewidth=line_width, markersize=1.5)
            # ax.fill_between(tlist, mean_costlist - std1_costlist, mean_costlist + std1_costlist, color=Traj_color_buf[i], alpha=0.15)
            # ax.fill_between(tlist, mean_costlist - std2_costlist, mean_costlist + std2_costlist, color=color_buf[i], alpha=0.1)
            # plt.plot(tlist, costlist, color_buf[i], markersize=2, label=exp_name)
            if Plot_suc_rate:
                ax.plot(tlist, sucRatelist, alpha=0.5, color=color_buf[i], linewidth=0.3, markersize=0.6)

            # ax.plot(tlist, mean_costlist, alpha=1.0, color=color_buf[i], marker=marker_buf[i], label=exp_name.capitalize(), linewidth=0.5, markersize=6)
            # ax.fill_between(tlist, mean_costlist - std1_costlist, mean_costlist + std1_costlist, color=color_buf[i], alpha=0.2)
            # ax.fill_between(tlist, mean_costlist - std2_costlist, mean_costlist + std2_costlist, color=color_buf[i], alpha=0.1)
            # # plt.plot(tlist, costlist, color_buf[i], markersize=2, label=exp_name)
            # # ax.plot(tlist, sucRatelist, alpha=0.5, color=color_buf[i], marker=marker_buf[i], linewidth=0.3, markersize=0.6)

        plot_fig(axList[0], timelist, costlist)
        plot_fig(axList[7], timelist, Total_costlist)
        plot_fig(axList[8], timelist, Vel_costlist)

    for i in range(len(exp_names)):
        print("[Final result] {}: SR: {} ({}/{}) , FR: {} ({}/{}) , ST: {} | # of files: {} \n".format(
            exp_names[i],
            SR[i]/float(exp_file_paths[i].__len__()), SR[i], float(exp_file_paths[i].__len__()),
            FR[i]/float(exp_file_paths[i].__len__()), FR[i], float(exp_file_paths[i].__len__()),
            ST[i],
            exp_file_paths[i].__len__()
        )
        )

if Plot_Result:
    exp_name = ["Pose Cost (position + rotation)",
                "Position_Cost (m)", "Rotation_Cost (rad)", "Vel_Cost (rad/s)", "Vel_Cost", "Acc_Cost", "Jerk_Cost",
                "Total Cost", "Vel_Cost (rad/s)"]
    for i in range(len(axList)):
        # axList[i].set_title("Optimization Progress (exp: {})".format(SCENE_NAME), fontsize=17)
        # axList[i].set_xlabel("time (sec)")
        axList[i].set_xticks(np.arange(0, (TORM_max_time if Plot_TORM else Trajopt_max_time)+0.00001, time_gap))
        axList[i].set_xlim([-(1 if Plot_TORM else 3), (TORM_max_time if Plot_TORM else Trajopt_max_time)+(1 if Plot_TORM else 3)])
        # axList[i].set_ylabel("{}".format(exp_name[i]), fontsize=17)
        plt.setp(axList[i].get_yticklabels(), visible=False)
        plt.setp(axList[i].get_xticklabels(), visible=False)
        axList[i].yaxis.set_ticks_position('none')
        if w_legend:
            axList[i].legend(loc='upper right',fontsize=fontsize_for_legend)
        axList[i].set_yscale("log")
        axList[i].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if set_grid:
            axList[i].grid(True, which='both')
        # axList[i].set_facecolor((1.0, 1.0, 1.0))

    for i in range(len(plotList)):
        plotList[i][0].tight_layout()
        # plotList[i][0].subplots_adjust(left=0.08, bottom=0.1, right=0.99, top=0.98, wspace=0.2, hspace=0.2)
        plotList[i][0].subplots_adjust(left=0.01, bottom=0.03, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
    plt.show()





# ax.grid(True, which='both', linestyle='-', linewidth='0.5', color='white')


# ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv', linewidth = 1.0)
# ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
# ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
# ax.legend(loc='best')
# ax.set_ylim([0.88,1.02])
# ax.set_ylabel("Accuracy")
# ax.set_xlabel("N_estimators")