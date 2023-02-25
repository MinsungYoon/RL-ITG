from copy import deepcopy
import numpy as np
from numpy import random as RD
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

import time
from tensorboardX import SummaryWriter
import utils
import itertools
import os

def check_grad_norm(net, TBoard, log_name, t):
    total_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** (1. / 2)
    TBoard.add_scalar(log_name, total_norm, t)


def sac(env, test_env, actor_critic=None, hidden_sizes=None, log_dir="", seed=1000, Boost_Q=0,
         steps_per_epoch=5000, epochs=2000, batch_size=1024, replay_size=int(1e6), gamma=0.99, polyak=0.995, alpha=0.2,
         start_use_network=7000, start_update_q=10000, start_update_pi=20000, update_every=500, update_steps=200,
         pi_lr=1e-4, q_lr=3e-4, pi_wd=1e-5, q_wd=5e-5, grad_clip=0.0, auto_entropy_tuning=False, ent_lr=3e-4, target_ent=0, w_act_reg=0.0,
         BN=False, DR=False, num_test_episodes=20, use_pretrained_model=False, log_path='', set_from_start=True):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("Cannot use GPU.")
        exit()

    TBoard = SummaryWriter(log_dir)

    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    print("[SAC] obs dim: {}, action dim: {}".format(obs_dim, act_dim))

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim, env.act_ll, env.act_ul, BN, DR, Boost_Q, hidden_sizes)
    ac_targ = deepcopy(ac)

    ac.eval()       # BN: use the calculat'ed' running mean and var in BN.
    ac_targ.eval()  # DR: use all activated network

    print("[MODEL] Actor_Critic: {}".format(ac))

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = utils.ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size), device=device)

    # Load Demonstration
    if use_pretrained_model:
        state_dict = torch.load(os.path.join(log_path, 'best_model.pt'))
        ac.load_state_dict(state_dict)
        ac_targ.load_state_dict(state_dict)
        print("[SAC] BC Model has been loaded.")

    # Count variables
    var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print("[MODEL] Count variables: {}".format(var_counts))

    #==================== Optimizer setting ====================
    pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr, weight_decay=pi_wd)

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = optim.Adam(q_params,  lr=q_lr,  weight_decay=q_wd)

    # pi_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(pi_optimizer, T_0=int(steps_per_epoch)*10, T_mult=1, eta_min=float(pi_lr*0.1), last_epoch=-1)
    # q_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(q_optimizer, T_0=int(steps_per_epoch)*10, T_mult=1, eta_min=float(q_lr*0.1), last_epoch=-1)

    if auto_entropy_tuning:
        target_entropy = target_ent # heuristic: act_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=ent_lr)

    #==================== Logging variables ====================
    log_ep_ret, log_ep_len, log_ep_task_suc, log_ep_dist_suc, log_ep_rot_suc, log_ep_imit_suc = [], [], [], [], [], []
    log_ep_col, log_ep_timeout, log_ep_earlystop = [], [], []
    if auto_entropy_tuning:
        log_alpha_loss, log_alpha_alpha, log_alpha_logalpha, log_alpha_logpi, log_alpha_targetent = [], [], [], [], []
    log_q_loss, log_q1, log_q2, log_q_reward, log_q_target, log_q_target_pi, log_q_target_alogpi = [], [], [], [], [], [], []
    log_pi_mu_mu, log_pi_mu_std, log_pi_std_mu, log_pi_std_std = [], [] ,[] ,[]
    log_pi_act_norm, log_pi_logp_pi, log_pi_alpha, log_pi_loss_pi = [], [], [], []

    #==================== SAC alpha-loss ====================
    def compute_loss_alpha(data):
        o = data['obs']
        pi, logp_pi, _, _ = ac.pi(o) # stocastic action (mu)
        alpha_loss = ( -log_alpha * (logp_pi - target_entropy).detach() ).mean()

        log_alpha_alpha.append(log_alpha.detach().exp().cpu().numpy())
        log_alpha_logalpha.append(log_alpha.detach().cpu().numpy())
        log_alpha_logpi.append(logp_pi.detach().mean().cpu().numpy())
        log_alpha_targetent.append(target_entropy)
        log_alpha_loss.append(alpha_loss.detach().cpu().numpy())

        return alpha_loss

    #==================== SAC Q-loss ====================
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _, _ = ac.pi(o2) # stocastic action

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)
            backup = backup.clamp_(-100, 100)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # print("Mean {} Max {} Min {}".format(((q - backup)**2).mean(), ((q - backup)**2).max(), ((q - backup)**2).min()))

        # Useful info for logging
        log_q_loss.append(loss_q)
        log_q1.append(q1.detach().mean().cpu().numpy())
        log_q2.append(q2.detach().mean().cpu().numpy())
        log_q_reward.append(r.detach().mean().cpu().numpy())
        log_q_target.append(backup.detach().mean().cpu().numpy())
        log_q_target_pi.append(q_pi_targ.detach().mean().cpu().numpy())
        log_q_target_alogpi.append((alpha * logp_a2).detach().mean().cpu().numpy())

        return loss_q

    #==================== SAC pi loss ====================
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi, mu, std = ac.pi(o) # stocastic action (mu)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        if w_act_reg == 0.0:
            loss_pi = (alpha * logp_pi - q_pi).mean()
        else:
            # print("Act reg!!!!!")
            act_reg = mu.norm(p=2, dim=1).mean()  # action norm regularization
            loss_pi = (alpha * logp_pi - q_pi).mean() + w_act_reg * act_reg

        log_pi_act_norm.append(pi.detach().abs().mean().cpu().numpy())
        log_pi_logp_pi.append(logp_pi.detach().mean().cpu().numpy())
        log_pi_alpha.append(alpha)
        log_pi_loss_pi.append(loss_pi.detach())
        log_pi_mu_mu.append(mu.detach().abs().mean())
        log_pi_mu_std.append(mu.detach().std())
        log_pi_std_mu.append(std.detach().abs().mean())
        log_pi_std_std.append(std.detach().std())

        return loss_pi

    #==================== Update ====================
    def update_Alpha(data):
        alpha_optimizer.zero_grad()
        loss_alpha = compute_loss_alpha(data)
        loss_alpha.backward()
        alpha_optimizer.step()

    def update_Q(data, t):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        check_grad_norm(ac.q1, TBoard, "GradNorm_Q1", t+1)
        check_grad_norm(ac.q2, TBoard, "GradNorm_Q2", t+1)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(ac.q1.parameters(), max_norm=grad_clip)
            torch.nn.utils.clip_grad_norm_(ac.q2.parameters(), max_norm=grad_clip)
        check_grad_norm(ac.q1, TBoard, "GradClipedNorm_Q1", t+1)
        check_grad_norm(ac.q2, TBoard, "GradClipedNorm_Q2", t+1)
        q_optimizer.step()

    def update_PI(data, t):
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        check_grad_norm(ac.pi, TBoard, "GradNorm_PI", t+1)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), max_norm=grad_clip)
        check_grad_norm(ac.pi, TBoard, "GradClipedNorm_PI", t+1)
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

    def update_target_network():
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for m, m_targ in zip(ac.modules(), ac_targ.modules()):
                if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.BatchNorm1d):
                    for p, p_targ in zip(m.parameters(), m_targ.parameters()): # weight & bias
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
                    if isinstance(m, torch.nn.BatchNorm1d): # running_mean & bias
                        for b, b_targ in zip(m.buffers(), m_targ.buffers()):
                            b_targ.copy_(b)

    #==================== Evaluation ====================
    def test_agent(tepoch):
        print("================ Evaluate agent ================")
        tlog_ep_ret, tlog_ep_len, tlog_ep_task_suc, tlog_ep_dist_suc, tlog_ep_rot_suc, tlog_ep_imit_suc = [], [], [], [], [], []
        tlog_ep_col, tlog_ep_timeout, tlog_ep_earlystop = [], [], []
        for ti in range(num_test_episodes):
            print("=== Eval {}-th ===".format(ti))
            # rollout
            to, td, tep_ret, tep_len = test_env.reset(set_from_start=True), False, 0.0, 0.0
            while not td:
                # Take deterministic actions at test time
                ta = get_action(to, deterministic=True)
                to, tr, td, tinfo = test_env.step(ta)
                tep_ret += tr
                tep_len += 1

            print("T:({:3}/{:3}) [Ep_ret] {:7.3f} [INFO] TS:{:2} ({:2}|{:2}) | IS:{:2} | C:{:2} | JR:{:2} | TO:{:2} | ES:{:2}".format(
                test_env.timestep, test_env.max_timestep, tep_ret,
                tinfo['task_suc_count'], tinfo['dist_suc_count'], tinfo['rot_suc_count'], tinfo['imitate_suc_count'],
                tinfo['is_col'], tinfo['is_jlimit'], tinfo['is_timeout'], tinfo['is_early_stop']
            ))

            test_env.visualize_solution()
            time.sleep(0.1)

            tlog_ep_ret.append(tep_ret)
            tlog_ep_len.append(tep_len/test_env.max_timestep)
            tlog_ep_task_suc.append(float(tinfo['task_suc_count'])/test_env.max_timestep)
            tlog_ep_dist_suc.append(float(tinfo['dist_suc_count'])/test_env.max_timestep)
            tlog_ep_rot_suc.append(float(tinfo['rot_suc_count'])/test_env.max_timestep)
            tlog_ep_imit_suc.append(float(tinfo['imitate_suc_count'])/test_env.max_timestep)
            tlog_ep_col.append(float(tinfo['is_col']))
            tlog_ep_timeout.append(float(tinfo['is_timeout']))
            tlog_ep_earlystop.append(float(tinfo['is_early_stop']))

        TBoard.add_scalar("Eval_EpRet",  sum(tlog_ep_ret)/len(tlog_ep_ret), tepoch)
        TBoard.add_scalar("Eval_EpLen",  sum(tlog_ep_len)/len(tlog_ep_len), tepoch)
        TBoard.add_scalar("Eval_EpSuc_Task", sum(tlog_ep_task_suc)/len(tlog_ep_task_suc), tepoch)
        TBoard.add_scalar("Eval_EpSuc_Dist", sum(tlog_ep_dist_suc)/len(tlog_ep_dist_suc), tepoch)
        TBoard.add_scalar("Eval_EpSuc_Rot",  sum(tlog_ep_rot_suc)/len(tlog_ep_rot_suc), tepoch)
        TBoard.add_scalar("Eval_EpSuc_Imit", sum(tlog_ep_imit_suc)/len(tlog_ep_imit_suc), tepoch)
        TBoard.add_scalar("Eval_EpEnd_Col", sum(tlog_ep_col)/len(tlog_ep_col), tepoch)
        TBoard.add_scalar("Eval_EpEnd_TO",  sum(tlog_ep_timeout)/len(tlog_ep_timeout), tepoch)
        TBoard.add_scalar("Eval_EpEnd_ES",  sum(tlog_ep_earlystop)/len(tlog_ep_earlystop), tepoch)

        print("================ Evaluate agent ================\n")
        return sum(tlog_ep_ret)/len(tlog_ep_ret)


    #==================== Action module ====================
    def get_action(o, deterministic=False): # default: stocastic action
        return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    #==================== Main Training Loop ====================
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    best_ep_ret = -np.inf

    o, ep_ret, ep_len = env.reset(set_from_start=set_from_start), 0.0, 0.0
    # Main loop: collect experience in env and update/log each epoch
    t = 0
    while t < total_steps:

        # select action
        if (t+1) >= start_use_network and RD.rand() > 0.05:
            a = get_action(o) # stocastic action
        else:
            a = env.random_action()

        # Step the env
        o2, r, d, info = env.step(a)

        # print("Act - reward: {}".format(reg_action * abs(a).sum()/7))

        ep_ret += r
        ep_len += 1

        if d:
            print("<Ep.Done> T:({:3}/{:3}) [Ep_ret] {:7.3f} [INFO] TS:{:2} ({:2}|{:2}) | IS:{:2} | C:{:2} | JR:{:2} | TO:{:2} | ES:{:2}".format(
                env.timestep, env.max_timestep, ep_ret,
                info['task_suc_count'], info['dist_suc_count'], info['rot_suc_count'], info['imitate_suc_count'],
                info['is_col'], info['is_jlimit'], info['is_timeout'], info['is_early_stop']
            ))

        d = False if info['is_timeout'] or info['is_early_stop'] else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # make sure to update most recent observation!
        o = o2

        # End of trajectory handling
        if info['is_col'] or info['is_timeout'] or info['is_early_stop']:
            log_ep_ret.append(ep_ret)
            log_ep_len.append(ep_len/env.max_timestep)
            log_ep_task_suc.append(float(info['task_suc_count'])/env.max_timestep)
            log_ep_dist_suc.append(float(info['dist_suc_count'])/env.max_timestep)
            log_ep_rot_suc.append(float(info['rot_suc_count'])/env.max_timestep)
            log_ep_imit_suc.append(float(info['imitate_suc_count'])/env.max_timestep)
            log_ep_col.append(float(info['is_col']))
            log_ep_timeout.append(float(info['is_timeout']))
            log_ep_earlystop.append(float(info['is_early_stop']))
            # env.visualize_demo()
            # time.sleep(10)
            # env.visualize_solution()
            # time.sleep(0.1)
            o, ep_ret, ep_len = env.reset(set_from_start=set_from_start), 0.0, 0.0


        # Update handling
        if (t+1) % update_every == 0 and ((t+1) >= start_update_q or (t+1) >= start_update_pi):
            print("\n(UH)[t+1:{:6}], Start update. ep_ret: {:7.3f} | ep_len {:4} | step_num ({:3}/{:3})\n".format(
                t+1, ep_ret, ep_len, env.timestep, env.max_timestep
            ))
            if BN or DR:    # BN: estimate running mean and var from data
                ac.train()  # DR: dropout neurons for regularization
            ac.to(device)
            ac_targ.to(device)
            if (t+1) >= start_update_pi:
                for ustep in range(update_steps):
                    batch = replay_buffer.sample_batch(batch_size)
                    # if auto_entropy_tuning and ustep % 5 == 0:
                    if auto_entropy_tuning:
                        log_alpha.requires_grad = True
                        update_Alpha(data=batch)
                        alpha = log_alpha.exp().cpu().item()
                        log_alpha.requires_grad = False
                    update_Q(data=batch, t=t)
                    update_PI(data=batch, t=t)
                    update_target_network()
                    # pi_scheduler.step()
                    # q_scheduler.step()
            else:
                for _ in range(update_steps):
                    batch = replay_buffer.sample_batch(batch_size)
                    update_Q(data=batch, t=t)
                    update_target_network()
                    # q_scheduler.step()
            if BN or DR:
                ac.eval()

            ac.to(torch.device('cpu'))
            ac_targ.to(torch.device('cpu'))

            TBoard.add_scalar('LR_Q', q_optimizer.param_groups[0]['lr'], t+1)
            TBoard.add_scalar('LR_PI', pi_optimizer.param_groups[0]['lr'], t+1)

            TBoard.add_scalar("Q_loss",     sum(log_q_loss)/len(log_q_loss)     ,t+1)
            TBoard.add_scalar("Q1",         sum(log_q1)/len(log_q1)             ,t+1)
            TBoard.add_scalar("Q2",         sum(log_q2)/len(log_q2)             ,t+1)
            TBoard.add_scalar("Q_reward",   sum(log_q_reward)/len(log_q_reward) ,t+1)
            TBoard.add_scalar("Q_target",   sum(log_q_target)/len(log_q_target) ,t+1)
            TBoard.add_scalar("Q_target_pi",sum(log_q_target_pi)/len(log_q_target_pi) ,t+1)
            TBoard.add_scalar("Q_target_alogpi",   sum(log_q_target_alogpi)/len(log_q_target_alogpi) ,t+1)
            if (t+1) >= start_update_pi:
                TBoard.add_scalar("PI_loss",    sum(log_pi_loss_pi)/len(log_pi_loss_pi)     ,t+1)
                TBoard.add_scalar("PI_alpha",   sum(log_pi_alpha)/len(log_pi_alpha)         ,t+1)
                TBoard.add_scalar("PI_logp_pi", sum(log_pi_logp_pi)/len(log_pi_logp_pi)     ,t+1)
                TBoard.add_scalar("PI_act_norm", sum(log_pi_act_norm)/len(log_pi_act_norm)  ,t+1)
                TBoard.add_scalar("PI_act_mu_mu", sum(log_pi_mu_mu)/len(log_pi_mu_mu)       ,t+1)
                TBoard.add_scalar("PI_act_mu_std", sum(log_pi_mu_std)/len(log_pi_mu_std)    ,t+1)
                TBoard.add_scalar("PI_act_std_mu", sum(log_pi_std_mu)/len(log_pi_std_mu)    ,t+1)
                TBoard.add_scalar("PI_act_std_std", sum(log_pi_std_std)/len(log_pi_std_std) ,t+1)

                if auto_entropy_tuning:
                    TBoard.add_scalar("Alpha_loss",     sum(log_alpha_loss)/len(log_alpha_loss)             ,t+1)
                    TBoard.add_scalar("Alpha_alpha",    sum(log_alpha_alpha)/len(log_alpha_alpha)           ,t+1)
                    TBoard.add_scalar("Alpha_logalpha", sum(log_alpha_logalpha)/len(log_alpha_logalpha)     ,t+1)
                    TBoard.add_scalar("Alpha_logpi",    sum(log_alpha_logpi)/len(log_alpha_logpi)           ,t+1)
                    TBoard.add_scalar("Alpha_targetent",sum(log_alpha_targetent)/len(log_alpha_targetent)   ,t+1)
                    print("LossQ:{:4.3f} LossPi:{:4.3f} LossAlpha:{:4.3f}\n".format(
                        sum(log_q_loss)/len(log_q_loss), sum(log_pi_loss_pi)/len(log_pi_loss_pi),
                        sum(log_alpha_loss)/len(log_alpha_loss)
                    ))
                else:
                    print("LossQ:{:4.3f} LossPi:{:4.3f}\n".format(
                        sum(log_q_loss)/len(log_q_loss), sum(log_pi_loss_pi)/len(log_pi_loss_pi),
                    ))
            if auto_entropy_tuning:
                log_alpha_loss, log_alpha_alpha, log_alpha_logalpha, log_alpha_logpi, log_alpha_targetent = [], [], [], [], []
            log_q_loss, log_q1, log_q2, log_q_reward, log_q_target, log_q_target_pi, log_q_target_alogpi  = [], [], [], [], [], [], []
            log_pi_act_norm, log_pi_logp_pi, log_pi_alpha, log_pi_loss_pi = [], [], [], []
            log_pi_mu_mu, log_pi_mu_std, log_pi_std_mu, log_pi_std_std = [], [], [], []

        # End of epoch handling
        if (t+1) >= start_update_pi and (t+1) % steps_per_epoch == 0:
            print("\n(EP)[t+1:{:6}], Epoch Logging. ep_ret: {:7.3f} | ep_len {:4} | step_num ({:3}/{:3})\n".format(
                t+1, ep_ret, ep_len, env.timestep, env.max_timestep
            ))
            epoch = (t+1) // steps_per_epoch

            # Log info about epoch
            TBoard.add_scalar("Epoch",              epoch                               ,epoch)
            TBoard.add_scalar("TotalEnvInteracts",  t                                   ,epoch)
            TBoard.add_scalar("EpRet",  sum(log_ep_ret)/len(log_ep_ret)                 ,epoch)
            TBoard.add_scalar("EpLen",  sum(log_ep_len)/len(log_ep_len)                 ,epoch)
            TBoard.add_scalar("EpSuc_Task", sum(log_ep_task_suc)/len(log_ep_task_suc)   ,epoch)
            TBoard.add_scalar("EpSuc_Dist", sum(log_ep_dist_suc)/len(log_ep_dist_suc)   ,epoch)
            TBoard.add_scalar("EpSuc_Rot",  sum(log_ep_rot_suc)/len(log_ep_rot_suc)     ,epoch)
            TBoard.add_scalar("EpSuc_Imit", sum(log_ep_imit_suc)/len(log_ep_imit_suc)   ,epoch)
            TBoard.add_scalar("EpEnd_Col", sum(log_ep_col)/len(log_ep_col)              ,epoch)
            TBoard.add_scalar("EpEnd_TO",  sum(log_ep_timeout)/len(log_ep_timeout)      ,epoch)
            TBoard.add_scalar("EpEnd_ES",  sum(log_ep_earlystop)/len(log_ep_earlystop)  ,epoch)
            log_ep_ret, log_ep_len, log_ep_task_suc, log_ep_dist_suc, log_ep_rot_suc, log_ep_imit_suc = [], [], [], [], [], []
            log_ep_col, log_ep_timeout, log_ep_earlystop = [], [], []

            elapsed_time = time.time()-start_time
            TBoard.add_scalar("Time_sec",   elapsed_time    ,epoch)
            TBoard.add_scalar("Time_min",   elapsed_time/60 ,epoch)

            # Test the performance of the deterministic version of the agent
            test_ep_ret = test_agent(epoch)

            # Save model
            if test_ep_ret >= best_ep_ret :
                best_ep_ret = test_ep_ret
                torch.save(ac.state_dict(), log_dir + '/best_model.pt')
                print("<<<<<<<<<<<<<<< Saved best_model.pt (ep: {}) >>>>>>>>>>>>>>>".format(epoch))
            torch.save(ac.state_dict(), log_dir + '/last_model.pt')
        t += 1
