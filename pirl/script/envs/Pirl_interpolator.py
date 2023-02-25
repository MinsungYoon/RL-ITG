import rospy
import numpy as np
import quaternion #(w, x, y, z)
import torch

from pirl_msgs.srv import fk

class Interpolator:
    def __init__(self, ac, start_conf, waypoints=[]):
        self.ac = ac
        self.start_conf = np.array(start_conf)
        self.waypoints = waypoints
        self.N_waypoints = len(waypoints)

        self.dt = 0.1

        self.fk_srv = rospy.ServiceProxy('/fk_solver', fk)

        self.ll = rospy.get_param("/robot/ll")
        self.ul = rospy.get_param("/robot/ul")
        self.n_dof = rospy.get_param("/robot/n_dof")
        self.c_joints = rospy.get_param("/robot/continuous_joints")

        self.cur_conf = self.start_conf
        self.prev_action = np.zeros(self.n_dof)
        self.goal_pos = None

        self.out_conf_list = [self.cur_conf]

    def get_action(self, o, deterministic=False): # default: stocastic action
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def execute(self, deterministic):
        for i in range(1, self.N_waypoints):
            self.goal_pos = np.array(self.waypoints[i])
            obs = self._get_obs()
            act = self.get_action(obs, deterministic=deterministic)
            self.cur_conf = self._set_action(act)
            self.prev_action = act
            self.out_conf_list.append(self.cur_conf)
        return self.out_conf_list

    def _get_obs(self):
        res = self.fk_srv(self.cur_conf.tolist())
        while not res.result:
            res = self.fk_srv(self.cur_conf.tolist())
        ee_pos = list(res.fk_result)
        diff = self._to_the_goal(ee_pos)
        observation = self.cur_conf.tolist() + self.prev_action.tolist() + ee_pos + self.goal_pos.tolist() + diff
        return np.array(observation)

    def _to_the_goal(self, ee_pos):
        goal_quat = np.quaternion(self.goal_pos[6], self.goal_pos[3], self.goal_pos[4], self.goal_pos[5])
        conj_cur_quat = np.quaternion(ee_pos[6], -1*ee_pos[3], -1*ee_pos[4], -1*ee_pos[5])
        diff_quat = goal_quat * conj_cur_quat
        diff = [
            self.goal_pos[0] - ee_pos[0],
            self.goal_pos[1] - ee_pos[1],
            self.goal_pos[2] - ee_pos[2],
            diff_quat.x,
            diff_quat.y,
            diff_quat.z,
            diff_quat.w
        ]
        return diff

    def _set_action(self, a):
        o2 = self.cur_conf + (a * self.dt)
        for j in self.c_joints:
            if o2[j] > np.pi:
                o2[j] -= 2*np.pi
            elif o2[j] < -np.pi:
                o2[j] += 2*np.pi
        o2 = np.clip(o2, self.ll, self.ul)
        return o2

    def fk_output(self):
        pos_list = []
        for conf in self.out_conf_list:
            res = self.fk_srv(conf)
            while not res.result:
                res = self.fk_srv(conf)
            pos_list.append(list(res.fk_result))
        return pos_list