import rospy
import numpy as np
import quaternion #(w, x, y, z)
from numpy import random as RD
from numpy import linalg as LA
import time
import sys
import os
import csv
import torch

from utils import PathVisualizer

from pirl_msgs.srv import collision
from pirl_msgs.srv import scene_set
from pirl_msgs.srv import scene_reset

from pirl_msgs.srv import fk
from pirl_msgs.srv import ik
from pirl_msgs.srv import rndvalconf
from pirl_msgs.srv import allfk
from pirl_msgs.srv import jaco

from vae_model import VanillaVAE

class PirlEnv_Fetch:
    def __init__(self, sceneType="", rewardType="", obsType="", t=None, vaeLogPath="", zDim=None, pObs=None, zStocastic=None, refineFlag=None):
        rospy.logwarn("[PirlTrajEnv_Fetch] Initialize PirlEnv")

        self.sceneType = sceneType
        self.rewardType = rewardType
        self.vaeLogPath = vaeLogPath
        self.z_dim = zDim
        self.p_obs = pObs
        self.z_stocastic = zStocastic
        self.refineFlag = refineFlag

        self.is_visualize = rospy.get_param("/use_rviz")
        if self.is_visualize:
            self.path_viz_ = PathVisualizer(is_test=False, is_second_path=True)

        self.cc_srv = rospy.ServiceProxy('/collision_check', collision)
        self.scene_set_srv = rospy.ServiceProxy('/scene_set', scene_set)
        self.scene_reset_srv = rospy.ServiceProxy('/scene_reset', scene_reset)

        self.ik_srv = rospy.ServiceProxy('/ik_solver', ik)
        self.fk_srv = rospy.ServiceProxy('/fk_solver', fk)
        self.allfk_srv = rospy.ServiceProxy('/allLinkfk_solver', allfk)
        self.jaco_srv = rospy.ServiceProxy('/jaco_reward', jaco)

        self.n_dof = rospy.get_param("/robot/n_dof")
        self.ll = rospy.get_param("/robot/ll")
        self.ul = rospy.get_param("/robot/ul")
        self.act_ll = rospy.get_param("/robot/vel_ll")
        self.act_ul = rospy.get_param("/robot/vel_ul")
        self.c_joints = rospy.get_param("/robot/continuous_joints")

        self.timestep = 0
        self.cumulated_episode_reward = 0
        self.task_suc_count, self.dist_suc_count, self.rot_suc_count, self.imitate_suc_count = 0, 0, 0, 0
        self.collision_flag = False
        self.joint_limit_flag = False
        self.mode = None

        self.path = []
        self.cur_conf, self.cur_pose = None, None
        self.prev_conf, self.prev_pose = None, None
        self.envLatent = None

        self.ll = np.array(self.ll) # 3.14... is continuous joint
        self.ul = np.array(self.ul)

        self.act_ll = np.array(list(self.act_ll))
        self.act_ul = np.array(list(self.act_ul))
        self.action_space_shape = (self.n_dof,)

        self.t = t
        self.obsType = obsType
        obsDim = 0
        if obsType == "onlyConf":
            obsDim += 7
        elif obsType == "Conf+LinkPoses":
            obsDim += 7 + 9 * 8
        elif obsType == "Conf+EE":
            obsDim += 7 + 9
        elif obsType == "onlyLinkPoses":
            obsDim += 9 * 8
        obsDim += 9 * t
        obsDim += self.z_dim
        if sceneType == "obs":
            self.vae = VanillaVAE(  in_channels=1,
                                    latent_dim=self.z_dim,
                                    hidden_dims=[32, 64, 128, 256, 512])
            state_dict = torch.load(vaeLogPath, map_location='cpu')
            self.vae.load_state_dict(state_dict)
            for param in self.vae.parameters():
                param.requires_grad = False
            if torch.cuda.is_available():
                self.vae.cuda()
            occ = torch.zeros(1, 1, 64, 64, 64)
            if torch.cuda.is_available():
                occ = occ.cuda()
            self.envFreeLatent = self.vae.encode_reparameterize(occ, stocastic=False)[0].tolist()
        self.observation_space_shape = (obsDim,)

        # [problem setting]
        self.BASE_DIR = "/data/torm_data/"
        self.free_problem_list = []
        self.obs_problem_list = []
        self.n_free_prob = None
        self.n_obs_prob = None
        self.prepare_problemSet()
        print("[ENV] # of problems = free: {}, obs: {}.".format(
            len(self.free_problem_list),
            len(self.obs_problem_list)
        ))

        self.gap_min = 1
        self.gap_max = 1

        self.demo_configs, self.target_poses, self.max_timestep = None, None, None

        self.threshold_early_stop_dist = 0.2  # 0.2 m
        self.threshold_dist_err = 0.04        # 4 cm
        self.threshold_rot_err = 0.104        # 6 deg
        self.threshold_imitate_err = 0.087    # 5 deg

        self.zero_task_dist = self.threshold_dist_err * 2   # 8 cm

        if rewardType == "Task+NullSpaceImitation":
            self.task_reward_flag = True
            self.null_space_imitate_reward_flag = True
            self.imitate_reward_flag = False
            self.task_suc_r = 10.0
            self.dist_suc_r = 1.0
            self.rot_suc_r = 1.0
            self.imit_suc_r = 0.0
        elif rewardType == "Task+Imitation":
            self.task_reward_flag = True
            self.null_space_imitate_reward_flag = False
            self.imitate_reward_flag = True
            self.task_suc_r = 10.0
            self.dist_suc_r = 1.0
            self.rot_suc_r = 1.0
            self.imit_suc_r = 0.0
        elif rewardType == "Task":
            self.task_reward_flag = True
            self.null_space_imitate_reward_flag = False
            self.imitate_reward_flag = False
            self.task_suc_r = 10.0
            self.dist_suc_r = 1.0
            self.rot_suc_r = 1.0
            self.imit_suc_r = 0.0
        elif rewardType == "Imitation":
            self.task_reward_flag = False
            self.null_space_imitate_reward_flag = False
            self.imitate_reward_flag = True
            self.task_suc_r = 0.0
            self.dist_suc_r = 0.0
            self.rot_suc_r = 0.0
            self.imit_suc_r = 10.0
        rospy.logwarn("[PirlEnv_Fetch] rewardType: %s (%d, %d)",
                      str(rewardType), self.task_reward_flag, self.imitate_reward_flag)

        self.dist_reward_params = [2.0, -65.6, 30.0, 1.0]
        self.rot_reward_params = [2.0, -5.0, 0.0, 0.0]
        self.imit_reward_params = [1.0, -3.8, 0.05, 1.0]
        self.null_space_imit_reward_params = [1.0, -15, 0.5, 1.0]

        self.w_col = 10.0
        self.w_jlimit = 1.0
        self.w_early_stop = 3.0

        self.act_reg = 20.0

        self.scene_reset_srv()

        rospy.logwarn("[PirlTrajEnv_Fetch] Action_space_shape: %s", str(self.action_space_shape))
        rospy.logwarn("[PirlTrajEnv_Fetch] Observation_space_shape: %s", str(self.observation_space_shape))
        rospy.logwarn("[PirlTrajEnv_Fetch] END init Env.")

    # ============================= set action and step =============================

    def _set_action(self, a):
        o2 = self.cur_conf + a
        for j in self.c_joints:
            if o2[j] > np.pi:
                o2[j] -= 2 * np.pi
            elif o2[j] < -np.pi:
                o2[j] += 2 * np.pi
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

        self.prev_conf = self.cur_conf
        self.prev_pose = self.cur_pose

        self._set_action(action)  # setup new cur_conf
        self.cur_pose = list(self.allfk_srv(self.cur_conf.tolist()).allfk_result)  # and cur_pose

        obs = self._get_obs()  # based on only cur_pose

        dist_reward, rot_reward, dist_err, rot_err = self._compute_task_reward()
        if not self.refineFlag:
            imitate_reward, imitate_err = self._compute_imitate_reward()

        null_space_imitate_reward = self._compute_null_space_imitate_reward()

        is_time_out = True if (self.timestep == self.max_timestep) else False
        is_col = True if self.collision_flag else False
        is_jlimit = True if self.joint_limit_flag else False

        is_early_stop = True if dist_err > self.threshold_early_stop_dist else False
        is_dist_suc = True if dist_err < self.threshold_dist_err else False
        is_rot_suc = True if rot_err < self.threshold_rot_err else False
        if not self.refineFlag:
            is_imitate_suc = True if imitate_err < self.threshold_imitate_err else False
        self.task_suc_count += (is_dist_suc * is_rot_suc)
        self.dist_suc_count += is_dist_suc
        self.rot_suc_count += is_rot_suc
        if not self.refineFlag:
            self.imitate_suc_count += is_imitate_suc

        reward = - self.w_col * is_col \
                 - self.w_jlimit * is_jlimit \
                 - self.w_early_stop * is_early_stop
                 # - self.act_reg * (LA.norm(action, 1)/self.n_dof)
        if self.task_reward_flag:
            reward += (
                    dist_reward
                    + self.dist_suc_r * is_dist_suc
                    + ( rot_reward if dist_err < self.zero_task_dist else 0 )
                    + ( self.rot_suc_r * is_rot_suc if dist_err < self.zero_task_dist else 0 )
                    + self.task_suc_r * (is_dist_suc * is_rot_suc)
                )
        if self.imitate_reward_flag and not self.refineFlag:
            reward += (
                    + imitate_reward
                    + self.imit_suc_r * is_imitate_suc
                )
        if self.null_space_imitate_reward_flag and not self.refineFlag:
            reward += (
                    + null_space_imitate_reward
            )

        # print("[REWARD] {:3.2f}: D[{:3.2f}] R[{:3.2f}] I[{:3.2f}] C[{:3.2f}] JL[{:3.2f}] ES[{:3.2f}] SD[{:3.2f}] SR[{:3.2f}] ST[{:3.2f}] SI[{:3.2f}]".format(
        #     reward, dist_reward, ( rot_reward if dist_err < self.zero_task_dist else 0 ), imitate_reward,
        #     -self.w_col * is_col, -self.w_jlimit * is_jlimit, -self.w_early_stop * is_early_stop,
        #     self.dist_suc_r * is_dist_suc, ( self.rot_suc_r * is_rot_suc if dist_err < self.zero_task_dist else 0 ),
        #     self.task_suc_r * (is_dist_suc * is_rot_suc), self.imit_suc_r * is_imitate_suc
        #     # - self.act_reg * (LA.norm(action, 1)/self.n_dof)
        # ))

        # analyze the results
        done = is_col or is_early_stop or is_time_out
        info = {
            'task_suc_count': self.task_suc_count,
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
        # robot state obs.
        if self.obsType == "onlyConf":
            obs += self.cur_conf.tolist()
        elif self.obsType == "Conf+LinkPoses":
            obs += self.cur_conf.tolist()
            for s in range(n_seg):
                obs += self.calc_pos_and_rot(self.cur_pose[7*s+0:7*s+7])
        elif self.obsType == "Conf+EE":
            obs += self.cur_conf.tolist()
            obs += self.calc_pos_and_rot(self.cur_pose[7*(n_seg-1)+0:7*(n_seg-1)+7])
        elif self.obsType == "onlyLinkPoses":
            for s in range(n_seg):
                obs += self.calc_pos_and_rot(self.cur_pose[7*s+0:7*s+7])

        # future target err.
        for i in range(self.t):
            t = self.timestep+(i+1)
            if t > self.max_timestep:
                t = self.max_timestep
            obs += self.calc_pos_and_rot_error(self.target_poses[t], self.cur_pose[7*(n_seg-1)+0:7*(n_seg-1)+7])

        # scene latent vector.
        obs += self.envLatent

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
    def reset(self, env_num=None, prob_path=None, mode='train', set_from_start=False):
        self.scene_reset_srv()

        self.mode = mode
        if env_num is None and prob_path is None:
            env_num, prob_path = self.sample_problem()
        self.demo_configs, self.target_poses, self.max_timestep = self.setup_problem(env_num, prob_path,
                                                                                     RD.random_integers(self.gap_min,
                                                                                                        self.gap_max),
                                                                                     set_from_start)
        if not self.refineFlag:
            for i, conf in enumerate(self.demo_configs):
                self.demo_configs[i] = self.refine_continuous_joints(conf)
            self.cur_conf = self.demo_configs[0]
        else:
            while True:
                self.cur_conf = np.array(self.ik_srv(self.target_poses[0]).ik_result)
                if self.cur_conf.size != 0:
                    break
                else:
                    env_num, prob_path = self.sample_problem()
                    self.demo_configs, self.target_poses, self.max_timestep = self.setup_problem(env_num, prob_path,
                                                                                                 RD.random_integers(self.gap_min,
                                                                                                                    self.gap_max),
                                                                                                 set_from_start)
        self.prev_conf = self.cur_conf
        self.cur_pose = list(self.allfk_srv(self.cur_conf.tolist()).allfk_result)
        self.prev_pose = self.cur_pose

        # for debugging
        self.path = [self.cur_conf]
        self.timestep = 0
        self.cumulated_episode_reward = 0
        self.task_suc_count = 0
        self.dist_suc_count = 0
        self.rot_suc_count = 0
        self.imitate_suc_count = 0
        self.collision_flag = False
        self.joint_limit_flag = False
        obs = self._get_obs()
        return obs

    def setup_problem(self, env_num, prob_path, gap, set_from_start):
        if env_num == -1:
            self.scene_reset_srv()
            if not self.refineFlag:
                file_configs = prob_path + "_config.csv"
                file_targetposes = prob_path + "_targetpose.csv"
            else:
                file_targetposes = prob_path + ".csv"
        else:
            self.scene_set_srv(env_num)
            n_scene = prob_path.split('/')[-2]
            n_prob = prob_path.split('/')[-1]
            file_configs = prob_path + "_config.csv"
            file_targetposes = prob_path[:prob_path.find('torm_solution')] + "problem/{}/prob_{}.csv".format(n_scene, n_prob)

        if self.sceneType == "obs":
            if env_num == -1:
                self.envLatent = self.envFreeLatent
            else:
                vae_name = self.BASE_DIR + "obs/scene_vae/vae_{}.npy".format(env_num)
                self.envLatent = np.load(vae_name).tolist()
        else:
            self.envLatent = [0.0] * self.z_dim

        if not self.refineFlag:
            configs = []
            with open(file_configs, 'r') as f:
                rdr = csv.reader(f)
                for i, line in enumerate(rdr):
                    if i % gap == 0:
                        configs.append([eval(ele) for ele in line])
        target_poses = []
        with open(file_targetposes, 'r') as f:
            rdr = csv.reader(f)
            for i, line in enumerate(rdr):
                if i % gap == 0:
                    if not self.refineFlag:
                        target_poses.append([eval(ele) for ele in line])
                    else:
                        buf = [ele.split(';') for ele in line]
                        target_poses.append([eval(buf[0][0]), eval(buf[1][0]), eval(buf[2][0]),
                                            eval(buf[2][1]), eval(buf[3][0]), eval(buf[4][0]), eval(buf[5][0])])
        total_len = len(target_poses)
        if RD.rand() > 0.2:
            start_idx = RD.random_integers(0, total_len-1-10)
        else:
            start_idx = 0
        if set_from_start:
            start_idx = 0
        end_idx = total_len-1
        max_timestep = end_idx - start_idx
        if not self.refineFlag:
            return np.array(configs[start_idx:end_idx+1]), np.array(target_poses[start_idx:end_idx+1]), max_timestep
        else:
            return None, np.array(target_poses[start_idx:end_idx+1]), max_timestep

    def sample_problem(self):
        if self.sceneType == "obs":
            if RD.rand() > self.p_obs:
                return -1, self.free_problem_list[RD.randint(0, self.n_free_prob)]
            else:  # obs problem
                path = self.obs_problem_list[RD.randint(0, self.n_obs_prob)]
                return int(path.split('/')[-2]), path
        else:
            return -1, self.free_problem_list[RD.randint(0, self.n_free_prob)]

    def prepare_problemSet(self):
        def get_problems(base_dir, buf_list):
            if not self.refineFlag:
                for dir_name in os.listdir(base_dir):
                    SUB_DIR = os.path.join(base_dir, dir_name)
                    if os.path.isdir(SUB_DIR):
                        for f_name in os.listdir(SUB_DIR):
                            DATA_FILE = os.path.join(SUB_DIR, f_name)
                            if os.path.isfile(DATA_FILE) and f_name.find("_config.csv") != -1:
                                buf_list.append(DATA_FILE[:DATA_FILE.find("_config.csv")])
            else:
                for f_name in os.listdir(base_dir):
                    DATA_FILE = os.path.join(base_dir, f_name)
                    if os.path.isfile(DATA_FILE):
                        buf_list.append(DATA_FILE[:DATA_FILE.find(".csv")])
        if not self.refineFlag:
            free_dir = self.BASE_DIR + "free/torm_solution"
        else:
            free_dir = self.BASE_DIR + "free/torm_solution/refine"
        get_problems(free_dir, self.free_problem_list)
        self.n_free_prob = len(self.free_problem_list)
        if self.sceneType == 'obs':
            if not self.refineFlag:
                obs_dir = self.BASE_DIR + "obs/torm_solution"
            else:
                obs_dir = self.BASE_DIR + "obs/torm_solution/refine"
            get_problems(obs_dir, self.obs_problem_list)
            self.n_obs_prob = len(self.obs_problem_list)






# ============================= compute_reward =============================
    def _compute_task_reward(self):
        target_pose = self.target_poses[self.timestep]
        cur_ee_pose = self.cur_pose[7*7+0:7*7+7]
        d_err = LA.norm(target_pose[:3]-cur_ee_pose[:3])

        tq = np.quaternion(target_pose[6], target_pose[3], target_pose[4], target_pose[5])
        conj_pq = np.quaternion(cur_ee_pose[6], -1*cur_ee_pose[3], -1*cur_ee_pose[4], -1*cur_ee_pose[5])
        diff_q = tq * conj_pq
        if abs(diff_q.w) > 1.0:
            r_err = 2*np.arccos(1.0)
        else:
            r_err = 2*np.arccos(abs(diff_q.w))

        dist_rwd = self.calc_reward(d_err, self.dist_reward_params)
        rot_rwd = self.calc_reward(r_err, self.rot_reward_params)

        # print("[REWARD] \ndist_err: [{}], \nrot_err: [{}]".format(log_dist_err, log_rot_err))
        # print("[REWARD] dist_err: [{:3.3f}], rot_err: [{:3.3f}]".format(d_err, r_err))
        return dist_rwd, rot_rwd, d_err, r_err

    def _compute_imitate_reward(self):
        ref_conf = self.demo_configs[self.timestep]
        conf_err = np.array(ref_conf) - self.cur_conf
        for c in self.c_joints:
            if conf_err[c] > np.pi:
                conf_err[c] -= 2 * np.pi
            elif conf_err[c] < -np.pi:
                conf_err[c] += 2 * np.pi
        imitate_err = LA.norm(conf_err, ord=1)/self.n_dof
        imitate_rwd = self.calc_reward(imitate_err, self.imit_reward_params)
        # print("[REWARD] log_imitate_err: [{:3.3f}]".format(imitate_err))
        return imitate_rwd, imitate_err

    def _compute_null_space_imitate_reward(self):
        ref_conf = self.demo_configs[self.timestep]
        conf_err = np.array(ref_conf) - self.cur_conf
        for c in self.c_joints:
            if conf_err[c] > np.pi:
                conf_err[c] -= 2 * np.pi
            elif conf_err[c] < -np.pi:
                conf_err[c] += 2 * np.pi
        objective_value = self.jaco_srv(conf_err.tolist(), self.cur_conf.tolist()).objective_value
        null_space_imitate_rwd = self.calc_reward(objective_value, self.null_space_imit_reward_params)
        # print("[REWARD] null_space_imitate_rwd: [{:3.3f}]".format(null_space_imitate_rwd))
        return null_space_imitate_rwd

    def calc_reward(self, err, params):
        return params[0]*np.exp(params[1]*err) - params[2]*(params[3]*err)**2

    # ============================= visualization =============================
    def visualize_solution(self):
        if self.is_visualize:
            self.path_viz_.clear()
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
        if self.is_visualize:
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

