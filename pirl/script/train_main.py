import os
import rospy
from envs.PirlEnv import PirlEnv_Fetch as PIRLEnv

import rospkg
class Args:
    def __init__(self, algo, exp, memo):
        self.algo = algo

        # [free|obs]
        self.sceneType = "free"
        # [Task|Imitation|Task+Imitation|Task+NullSpaceImitation]
        self.rewardType = "Task+NullSpaceImitation"
        # [onlyConf|Conf+LinkPoses|Conf+EE|onlyLinkPoses]
        self.obsType = "Conf+LinkPoses"
        self.t = 1

        # Training
        self.seed = 123124
        self.replay_size = int(1e6)
        self.gamma = 0.99
        self.polyak = 0.995

        if self.algo == 'sac':
            self.alpha = 0.0
            self.auto_entropy_tuning = True
            self.ent_lr = 1e-4
            self.target_ent = 3

            self.use_pretrained_model = False
            self.log_path = "/data/pirl_network/logs/FREEALL/R10-1-1P10-1-3_free_Task_Conf+LinkPoses_T1_ep_100000000_bs_4096_plr_7e-05_qlr_0.0001_pwd_0.0001_qwd_0.0001_gclip_10.0_hids_[1024, 1024, 1024]_BQ_0_BN_False_DR_False_AET_True_elr_0.0001_TEnt_3_AReg_3_SFS_False"

            if self.sceneType == "obs":
                self.p_obs = 0.6
                self.z_stocastic = False
                self.z_dim = 32
                self.vae_log_path = "/data/pirl_network/VAE/OnlyTable_ep_500_steps5_lr_0.0001_bs_64_z_dim32_WD_1e-06_wKLD_MAX_5e-06_start0.0_duration100.0_cudaTrue" \
                                    + "/VAE_best.pkl"
            else:
                self.p_obs = 0.0
                self.z_stocastic = None
                self.z_dim = 0
                self.vae_log_path = ""


        self.steps_per_epoch = 10000
        self.epochs = 100000000
        self.batch_size = 4096

        if not self.use_pretrained_model:
            self.start_use_network = 1000
            self.start_update_q = 2000
            self.start_update_pi = 2000
        else:
            self.start_use_network = 0
            self.start_update_q = 10000
            self.start_update_pi = 10000
        self.update_every = 2000
        self.update_steps = 400

        self.num_test_episodes = 50

        # Algo
        if self.algo == 'ddpg':
            self.act_noise = 0.1  # ~5.73 degree
        self.BN = False
        self.DR = False
        self.hidden_sizes = [1024, 1024, 1024]
        self.Boost_Q = 0

        self.w_act_reg = 3

        self.pi_lr = 7e-05
        self.q_lr  = 1e-04
        self.pi_wd = 1e-04
        self.q_wd  = 1e-04
        self.grad_clip = 10.0

        self.set_from_start = False

        self.exp_name = "{}_{}_{}_T{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.sceneType, self.rewardType, self.obsType, self.t,
            "ep", self.epochs,
            "bs", self.batch_size,
            "plr", self.pi_lr,
            "qlr", self.q_lr,
            "pwd", self.pi_wd,
            "qwd", self.q_wd,
            "gclip", self.grad_clip,
            "hids", self.hidden_sizes,
            "BQ", self.Boost_Q,
            "BN", self.BN,
            "DR", self.DR,
            "AET", self.auto_entropy_tuning,
            "elr", self.ent_lr,
            "TEnt", self.target_ent,
            "AReg", self.w_act_reg,
            "SFS", self.set_from_start,
            "ZDim", self.z_dim,
            "PO", self.p_obs,
            "ZS", self.z_stocastic,
            "RF", self.use_pretrained_model
        )
        if memo:
            self.exp_name = memo + "_" + self.exp_name
        if exp:
            self.exp_name = exp + "/" + self.exp_name

        rospack = rospkg.RosPack()
        self.log_dir = rospack.get_path('pirl') + "/script/logs/" + self.algo + "/" + self.exp_name
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        rospy.logwarn("[===INFO===] Logdir: "+self.log_dir)

def launch(args):
    env = PIRLEnv(sceneType=args.sceneType, rewardType=args.rewardType, obsType=args.obsType, t=args.t,
                  vaeLogPath=args.vae_log_path, zDim=args.z_dim, pObs=args.p_obs, zStocastic=args.z_stocastic, refineFlag=args.use_pretrained_model)
    test_env = PIRLEnv(sceneType=args.sceneType, rewardType=args.rewardType, obsType=args.obsType, t=args.t,
                       vaeLogPath=args.vae_log_path, zDim=args.z_dim, pObs=args.p_obs, zStocastic=args.z_stocastic, refineFlag=args.use_pretrained_model)
    if args.algo == "ddpg":
        rospy.logwarn("[===INFO===] start DDPG algo.")
        from algos.ddpg import ddpg
        from algos.ddpg import core
        ddpg.ddpg(env, test_env, core.MLPActorCritic, args.hidden_sizes, args.log_dir, args.seed, args.Boost_Q,
                  args.steps_per_epoch, args.epochs, args.batch_size, args.replay_size, args.gamma, args.polyak,
                  args.start_use_network, args.start_update_q, args.start_update_pi, args.update_every, args.update_steps,
                  args.pi_lr, args.q_lr, args.pi_wd, args.q_wd, args.grad_clip,
                  args.BN, args.DR, args.act_noise, args.num_test_episodes)

    elif args.algo == 'sac':
        rospy.logwarn("[===INFO===] start SAC algo.")
        from algos.sac import sac
        from algos.sac import core
        sac.sac(env, test_env, core.MLPActorCritic, args.hidden_sizes, args.log_dir, args.seed, args.Boost_Q,
                  args.steps_per_epoch, args.epochs, args.batch_size, args.replay_size, args.gamma, args.polyak, args.alpha,
                  args.start_use_network, args.start_update_q, args.start_update_pi, args.update_every, args.update_steps,
                  args.pi_lr, args.q_lr, args.pi_wd, args.q_wd, args.grad_clip, args.auto_entropy_tuning, args.ent_lr, args.target_ent, args.w_act_reg,
                  args.BN, args.DR, args.num_test_episodes, args.use_pretrained_model, args.log_path, args.set_from_start)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--memo', type=str, default='')
    arg_parser = parser.parse_args()

    rospy.init_node("train_main")
    rospy.logwarn("[===INFO===] =======> train_main")

    args = Args(arg_parser.algo, arg_parser.exp, arg_parser.memo)

    launch(args)
