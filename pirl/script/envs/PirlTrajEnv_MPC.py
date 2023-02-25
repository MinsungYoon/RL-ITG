import rospy
import numpy as np
import quaternion #(w, x, y, z)
from numpy import random as RD
from numpy import linalg as LA
import time
import sys
import os
import csv

from utils import PathVisualizer

from pirl_msgs.srv import collision
from pirl_msgs.srv import scene_set
from pirl_msgs.srv import scene_reset

from pirl_msgs.srv import fk
from pirl_msgs.srv import ik
from pirl_msgs.srv import rndvalconf
from pirl_msgs.srv import allfk

class PirlTrajEnv_Fetch:
    def __init__(self):
        rospy.logwarn("[PirlTrajEnv_Fetch] Initialize PirlEnv")
        self.path_viz_ = PathVisualizer(is_test=False, is_second_path=True)

        self.cc_srv = rospy.ServiceProxy('/collision_check', collision)
        self.scene_set_srv = rospy.ServiceProxy('/scene_set', scene_set)
        self.scene_reset_srv = rospy.ServiceProxy('/scene_reset', scene_reset)

        self.fk_srv = rospy.ServiceProxy('/fk_solver', fk)
        self.allfk_srv = rospy.ServiceProxy('/allLinkfk_solver', allfk)
        self.ik_srv = rospy.ServiceProxy('/ik_solver', ik)
        self.rndvalconf_srv = rospy.ServiceProxy('/rndvalconf', rndvalconf)

        self.n_dof = rospy.get_param("/robot/n_dof")
        self.ll = rospy.get_param("/robot/ll")
        self.ul = rospy.get_param("/robot/ul")
        self.act_ll = rospy.get_param("/robot/vel_ll")
        self.act_ul = rospy.get_param("/robot/vel_ul")
        self.c_joints = rospy.get_param("/robot/continuous_joints")

        self.timestep = 0
        self.cumulated_episode_reward = 0
        self.dist_suc_count, self.rot_suc_count, self.imitate_suc_count = 0, 0, 0
        self.collision_flag = False
        self.joint_limit_flag = False

        self.path = []
        self.cur_conf, self.cur_pose = None, None
        self.prev_confs, self.prev_poses = None, None

        self.ll = np.array(self.ll) # 3.14... is continuous joint
        self.ul = np.array(self.ul)

        self.act_ll = np.array(list(self.act_ll) * 6)
        self.act_ul = np.array(list(self.act_ul) * 6)
        self.action_space_shape = (self.n_dof * 6,)
        self.observation_space_shape = (268,)

        # problem setting
        self.t = 6
        self.BASE_DIR = "/home/minsungyoon/nvme/torm_data/suc/"
        self.n_waypoints = [4, 5, 6, 7]
        self.n_get_train_data_start = [0, 0, 0, 0]
        self.n_get_train_data_end = [900, 900, 900, 400]
        self.n_get_test_data_start = [900, 900, 900, 400]
        self.n_get_test_data_end = [1000, 1000, 1000, 500]

        self.gap_min = 1
        self.gap_max = 3

        self.demo_configs, self.target_poses, self.max_timestep = None, None, None

        self.threshold_early_stop_dist = 1 # 1 m
        self.threshold_dist_err = 0.1 # 10 cm
        self.threshold_rot_err = 0.17 # 10 deg
        self.threshold_imitate_err = 0.3

        self.w_task_dist = 2
        self.wexp_task_dist = self.threshold_dist_err * 5

        self.w_task_rot = 1
        self.wexp_task_rot = self.threshold_rot_err * 5

        self.w_imitate = 5
        self.wexp_imitate = self.threshold_imitate_err

        self.w_alpha = 0.8

        self.w_col = 2
        self.w_jlimit = 1
        self.w_early_stop = 3



        rospy.logwarn("[PirlTrajEnv_Fetch] action_space_shape: %s", str(self.action_space_shape))
        rospy.logwarn("[PirlTrajEnv_Fetch] observation_space_shape: %s", str(self.observation_space_shape))
        rospy.logwarn("[PirlTrajEnv_Fetch] END init Env.")

    @staticmethod
    def queue_push(queue, item):
        queue.pop(0)
        queue.append(item)

    # ============================= set action and step =============================

    def _set_action(self, a):
        o2 = self.cur_conf + a
        for j in self.c_joints:
            if o2[j] > np.pi:
                o2[j] -= 2*np.pi
            elif o2[j] < -np.pi:
                o2[j] += 2*np.pi
        joint_limit_cliped_o2 = np.clip(o2, self.ll, self.ul)

        if self.mode == 'train':
            no_out_of_region = np.allclose(o2, joint_limit_cliped_o2)
            if no_out_of_region:
                res = self.cc_srv(joint_limit_cliped_o2.tolist())
                if res.collision_result:  # collision
                    self.collision_flag = True
                else:
                    self.cur_conf = joint_limit_cliped_o2
            else:  # joint limit violation
                self.joint_limit_flag = True
        else:
            no_out_of_region = np.allclose(o2, joint_limit_cliped_o2)
            self.cur_conf = joint_limit_cliped_o2
            if not no_out_of_region:
                print("Joint Limit! t: {}".format(self.timestep))

    def step(self, action):
        self.timestep += 1

        self.queue_push(self.prev_confs, self.cur_conf)
        self.queue_push(self.prev_poses, self.cur_pose)

        self._set_action(action[0:7])  # setup new cur_conf
        self.cur_pose = list(self.allfk_srv(self.cur_conf.tolist()).allfk_result)  # and cur_pose

        obs = self._get_obs()  # based on only cur_pose

        predicted_confs = []
        for i in range(self.t):
            predicted_confs.append(self.prev_confs[-1] + action[i*7+0: i*7+7])
        dist_reward, rot_reward, log_dist_err, log_rot_err = self._compute_task_reward(predicted_confs)
        imitate_reward, log_imitate_err = self._compute_imitate_reward(predicted_confs)

        is_time_out = True if (self.timestep == self.max_timestep) else False
        is_col = True if self.collision_flag else False
        is_jlimit = True if self.joint_limit_flag else False

        is_early_stop = True if log_dist_err[0] > self.threshold_early_stop_dist else False
        is_dist_suc = True if log_dist_err[0] < self.threshold_dist_err else False
        is_rot_suc = True if log_rot_err[0] < self.threshold_rot_err else False
        is_imitate_suc = True if log_imitate_err[0] < self.threshold_imitate_err else False
        self.dist_suc_count += is_dist_suc
        self.rot_suc_count += is_rot_suc
        self.imitate_suc_count += is_imitate_suc

        reward = dist_reward + rot_reward + imitate_reward \
                 - self.w_col * is_col - self.w_jlimit * is_jlimit - self.w_early_stop * is_early_stop

        # print("[REWARD]       R[{:3.3f}] = D[{:3.3f}] R[{:3.3f}] I[{:3.3f}] C[{:3.3f}] JL[{:3.3f}]".format(
        #     reward, dist_reward, rot_reward, imitate_reward, -self.w_col*is_col, -self.w_jlimit*is_jlimit
        # ))

        # analyze the results
        done = is_col or is_early_stop or is_time_out or is_jlimit
        info = {
            'dist_suc_count': self.dist_suc_count,
            'rot_suc_count': self.rot_suc_count,
            'imitate_suc_count': self.imitate_suc_count,
            'is_col': is_col,
            'is_jlimit': is_jlimit,
            'is_early_stop': is_early_stop,
            'is_timeout': is_time_out
        }

        # for debugging
        self.cumulated_episode_reward += reward
        self.collision_flag = False
        self.joint_limit_flag = False
        self.path.append(self.cur_conf)
        return obs, reward, done, info

    # ============================= get observation =============================
    def _get_obs(self):
        n_seg = int(len(self.cur_pose)/7)

        obs = []
        # conf
        obs += self.cur_conf.tolist()
        # all link and vel
        for s in range(n_seg):
            obs += self.calc_pos_and_rot(self.cur_pose[7*s+0:7*s+7])
            obs += self.calc_pos_and_rot_error(self.cur_pose[7*s+0:7*s+7], self.prev_poses[-1][7*s+0:7*s+7])

        for p in range(-self.t, 0): # past and current errors with target points
            p_idx = self.timestep + p
            if p_idx < 0:
                p_idx = 0
            obs += self.calc_pos_and_rot_error(self.target_poses[p_idx], self.prev_poses[p][7*(n_seg-1)+0:7*(n_seg-1)+7])

        obs += self.calc_pos_and_rot_error(self.target_poses[self.timestep], self.cur_pose[7*(n_seg-1)+0:7*(n_seg-1)+7])

        for i in range(self.t):  # future
            t = self.timestep+(i+1)
            if t > self.max_timestep:
                t = self.max_timestep
            obs += self.calc_pos_and_rot_error(self.target_poses[t], self.cur_pose[7*(n_seg-1)+0:7*(n_seg-1)+7])

        return np.array(obs)

    @staticmethod
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

    @staticmethod
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

    # ============================= reset and problem setting =============================
    def reset(self, w=None, p=None, gap=None, mode='train', set_from_start=False):
        self.mode = mode
        if w is None and p is None and gap is None:
            w, p, gap = self.get_random_problem(mode)
        # print("\n[RESET] [w: {}], [p:{}]\n".format(w, p))
        self.demo_configs, self.target_poses, self.max_timestep = self.setup_problem(w, p, gap, set_from_start)
        for i, conf in enumerate(self.demo_configs):
            self.demo_configs[i] = self.refine_continuous_joints(conf)
        self.cur_conf = self.demo_configs[0]
        self.prev_confs = [self.cur_conf] * self.t
        self.cur_pose = list(self.allfk_srv(self.cur_conf.tolist()).allfk_result)
        self.prev_poses = [self.cur_pose] * self.t

        # for debugging
        self.path = [self.cur_conf]
        self.timestep = 0
        self.cumulated_episode_reward = 0
        self.dist_suc_count = 0
        self.rot_suc_count = 0
        self.imitate_suc_count = 0
        self.collision_flag = False
        self.joint_limit_flag = False
        obs = self._get_obs()
        return obs

    def setup_problem(self, w, p, gap, set_from_start):
        data_dir = self.BASE_DIR + str(w)
        file_configs = os.path.join(data_dir, "{}_config.csv".format(p))
        file_targetposes = os.path.join(data_dir, "{}_targetpose.csv".format(p))
        configs = []
        target_poses = []
        with open(file_configs, 'r') as f:
            rdr = csv.reader(f)
            for i, line in enumerate(rdr):
                if i % gap == 0:
                    configs.append([eval(ele) for ele in line])
        with open(file_targetposes, 'r') as f:
            rdr = csv.reader(f)
            for i, line in enumerate(rdr):
                if i % gap == 0:
                    target_poses.append([eval(ele) for ele in line])
        total_len = len(configs)
        if RD.rand() > 0.4:
            start_idx = RD.random_integers(0, total_len-1-10)
        else:
            start_idx = 0
        if set_from_start:
            start_idx = 0
        end_idx = total_len-1
        max_timestep = end_idx - start_idx
        return np.array(configs[start_idx:end_idx+1]), np.array(target_poses[start_idx:end_idx+1]), max_timestep

    def get_random_problem(self, mode):
        w_idx = RD.random_integers(0, len(self.n_waypoints)-1)
        if mode == 'train':
            p = RD.random_integers(self.n_get_train_data_start[w_idx], self.n_get_train_data_end[w_idx]-1)
        else:
            p = RD.random_integers(self.n_get_test_data_start[w_idx], self.n_get_test_data_end[w_idx]-1)
        gap = RD.random_integers(self.gap_min, self.gap_max)
        return self.n_waypoints[w_idx], p, gap

    # ============================= compute_reward =============================
    def _compute_task_reward(self, predicted_confs):
        predicted_ee_pose_list = []
        target_pose_list = []
        for i in range(self.t):
            predicted_ee_pose_list.append(self.fk_srv(predicted_confs[i].tolist()).fk_result)
            t = self.timestep + i
            if t > self.max_timestep:
                t = self.max_timestep
            target_pose_list.append(self.target_poses[t])
        log_dist_err = []
        log_rot_err = []
        dist_rwd = 0.0
        rot_rwd = 0.0
        for t, (target, predict) in enumerate(zip(target_pose_list, predicted_ee_pose_list)):
            d_err = LA.norm(target[:3]-predict[:3])
            log_dist_err.append(d_err)
            dist_rwd += (self.w_alpha ** t) * np.exp(-d_err/self.wexp_task_dist)

            tq = np.quaternion(target[6], target[3], target[4], target[5])
            conj_pq = np.quaternion(predict[6], -1*predict[3], -1*predict[4], -1*predict[5])
            diff_q = tq * conj_pq
            if abs(diff_q.w) > 1.0:
                r_err = 2*np.arccos(1.0)
            else:
                r_err = 2*np.arccos(abs(diff_q.w))
            log_rot_err.append(r_err)
            rot_rwd += (self.w_alpha ** t) * np.exp(-r_err/self.wexp_task_rot)

        # print("[REWARD] \ndist_err: [{}], \nrot_err: [{}]".format(log_dist_err, log_rot_err))
        # print("[REWARD] dist_err: [{:3.3f}], rot_err: [{:3.3f}]".format(dist_err, rot_err))
        return self.w_task_dist * (dist_rwd / self.t), self.w_task_rot * (rot_rwd / self.t), log_dist_err, log_rot_err

    def _compute_imitate_reward(self, predicted_confs):
        imitate_rwd = 0.0
        log_imitate_err = []
        for i in range(self.t):
            t = self.timestep+i
            if t > self.max_timestep:
                t = self.max_timestep
            imitate_err = LA.norm(predicted_confs[i] - np.array(self.demo_configs[t]), ord=1)
            log_imitate_err.append(imitate_err)
            imitate_rwd += (self.w_alpha ** i) * np.exp(-imitate_err/self.wexp_imitate)
        # print("[REWARD] log_imitate_err: [{}]".format(log_imitate_err))
        return self.w_imitate * (imitate_rwd / self.t), log_imitate_err

    # ============================= visualization =============================
    def visualize_solution(self):
        # target poses
        self.path_viz_.pub_eepath_strip(self.target_poses, False)
        self.path_viz_.pub_eepath_arrow(self.target_poses, False)

        # current solution
        self.path_viz_.pub_path(self.path)
        eepath = []
        for conf in self.path:
            eepath.append(list(self.fk_srv(conf.tolist()).fk_result))
        self.path_viz_.pub_eepath_strip(eepath, True)
        self.path_viz_.pub_eepath_arrow(eepath, True)

    def visualize_demo(self):
        # target poses
        self.path_viz_.pub_eepath_strip(self.target_poses, False)
        self.path_viz_.pub_eepath_arrow(self.target_poses, False)

        # current solution
        self.path_viz_.pub_path(self.demo_configs)
        eepath = []
        for conf in self.demo_configs:
            eepath.append(list(self.fk_srv(conf.tolist()).fk_result))
        self.path_viz_.pub_eepath_strip(eepath, True)
        self.path_viz_.pub_eepath_arrow(eepath, True)

    def random_action(self):
        return RD.uniform(self.act_ll, self.act_ul)

    def refine_continuous_joints(self, conf):
        if self.c_joints:
            for j in self.c_joints:
                if conf[j] > np.pi:
                    buf = conf[j] - int(conf[j]/(2*np.pi))*2*np.pi
                    if buf > np.pi:
                        buf -= 2*np.pi
                    conf[j] = buf
                elif conf[j] < -np.pi:
                    buf = conf[j] - int(conf[j]/(2*np.pi))*2*np.pi
                    if buf < -np.pi:
                        buf += 2*np.pi
                    conf[j] = buf
        return conf