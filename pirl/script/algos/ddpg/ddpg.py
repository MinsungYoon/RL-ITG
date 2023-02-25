from copy import deepcopy
import numpy as np
from numpy import random as RD
import torch
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
import utils
import rospy


def ddpg(env, test_env, actor_critic=None, hidden_sizes=None, log_dir="", seed=1000, Boost_Q=0,
         steps_per_epoch=5000, epochs=2000, batch_size=1024, replay_size=int(1e6), gamma=0.99, polyak=0.995,
         start_use_network=7000, start_update_q=10000, start_update_pi=20000, update_every=500, update_steps=200,
         pi_lr=1e-4, q_lr=3e-4, pi_wd=1e-5, q_wd=5e-5, grad_clip=0.0,
         BN=False, DR=False, act_noise=0.1, num_test_episodes=20):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        exit()

    TBoard = SummaryWriter(log_dir)

    obs_dim = env.observation_space_shape[0]
    act_dim = env.action_space_shape[0]
    print("[DDPG] obs dim: {}, action dim: {}".format(obs_dim, act_dim))

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim, env.act_ll, env.act_ul, BN, DR, Boost_Q, hidden_sizes)
    ac.apply(utils.init_weights)
    ac_targ = deepcopy(ac)
    if BN:
        ac.eval() # use the calculat'ed' running mean and var in BN.
        ac_targ.eval()
    else:
        ac.train() # always activate DR.
        ac_targ.train()
    print("[MODEL] Actor_Critic: {}".format(ac))

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = utils.ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(replay_size), device=device)

    # Count variables
    var_counts = tuple(utils.count_vars(module) for module in [ac.pi, ac.q])
    print("[MODEL] Count variables: {}".format(var_counts))

    #==================== Optimizer setting ====================
    pi_optimizer = optim.Adam(ac.pi.parameters(), lr=pi_lr, weight_decay=pi_wd)
    q_optimizer  = optim.Adam(ac.q.parameters(),  lr=q_lr,  weight_decay=q_wd )

    pi_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(pi_optimizer, T_0=int(steps_per_epoch/2), T_mult=1, eta_min=float(pi_lr*0.001), last_epoch=-1)
    q_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(q_optimizer, T_0=int(steps_per_epoch/2), T_mult=1, eta_min=float(q_lr*0.001), last_epoch=-1)

    #==================== Logging variables ====================
    log_ep_ret, log_ep_len = [], []
    log_loss_q, log_loss_pi, log_q_vals, log_q_target, log_q_reward, log_q2 = [], [], [], [], [], []

    #==================== DDPG Q-loss ====================
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            q_2 = gamma * (1 - d) * q_pi_targ
            backup = r + q_2

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # print("Mean {} Max {} Min {}".format(((q - backup)**2).mean(), ((q - backup)**2).max(), ((q - backup)**2).min()))

        # Useful info for logging
        log_loss_q.append(loss_q)
        log_q_vals.append(q.mean().cpu().detach().numpy())
        log_q_target.append(backup.mean().cpu().detach().numpy())
        log_q_reward.append(r.mean().cpu().detach().numpy())
        log_q2.append(q_2.mean().cpu().detach().numpy())

        return loss_q

    #==================== DDPG pi loss ====================
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    #==================== Update ====================
    def update_Q(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(ac.q.parameters(), max_norm=grad_clip)
        q_optimizer.step()

    def update_PI(data):
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), max_norm=grad_clip)
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        log_loss_pi.append(loss_pi)

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
    def test_agent(mode, eph):
        print("================ Testing agent (mode:{}) ================".format(mode))
        tlog_ep_ret, tlog_ep_len = [], []
        n_suc, n_col, n_timeout = 0.0, 0.0, 0.0
        for i in range(num_test_episodes):
            # rollout
            to, td, tep_ret, tep_len = test_env.reset(), False, 0.0, 0.0
            while not td:
                # Take deterministic actions at test time
                ta = get_action(to)
                to, tr, td, tinfo = test_env.step(ta)
                tep_ret += tr
                tep_len += 1
                if test_env.demo and (i == num_test_episodes-1):
                    print("[TEST] Action ======================= [{:3.3f}] [{:3.3f}] [{:3.3f}] [{:3.3f}] [{:3.3f}] [{:3.3f}] [{:3.3f}]".format(
                        ta[0], ta[1], ta[2], ta[3], ta[4], ta[5], ta[6]
                    ))
                    print("[TEST] Reward: {:4.3f} S:{:4} | C:{:4} | TO:{:4}".format(
                        tr, tinfo['is_suc'], tinfo['is_col'], tinfo['is_timeout']
                    ))
            if test_env.demo and (i == num_test_episodes-1):
                test_env.visualize_demo()
            n_suc       += tinfo['is_suc']
            n_col       += tinfo['is_col']
            n_timeout   += tinfo['is_timeout']

            tlog_ep_ret.append(tep_ret)
            tlog_ep_len.append(tep_len)

        if mode == 'train':
            TBoard.add_scalar("EvalTrainNSuc", float(n_suc)/num_test_episodes,eph)
            TBoard.add_scalar("EvalTrainNCol", float(n_col)/num_test_episodes,eph)
            TBoard.add_scalar("EvalTrainNTimeout", float(n_timeout)/num_test_episodes,eph)
            TBoard.add_scalar("EvalTrainEpRet", sum(tlog_ep_ret)/len(tlog_ep_ret),eph)
            TBoard.add_scalar("EvalTrainEpLen", sum(tlog_ep_len)/len(tlog_ep_len),eph)
            return 0.0
        else:
            print("================[TEST] Suc|Col|Timeout: [{} | {} | {}] (total: {})\n".format(
                n_suc,n_col,n_timeout,num_test_episodes
            ))
            TBoard.add_scalar("EvalTestNSuc", float(n_suc)/num_test_episodes,eph)
            TBoard.add_scalar("EvalTestNCol", float(n_col)/num_test_episodes,eph)
            TBoard.add_scalar("EvalTestNTimeout", float(n_timeout)/num_test_episodes,eph)
            TBoard.add_scalar("EvalTestEpRet", sum(tlog_ep_ret)/len(tlog_ep_ret),eph)
            TBoard.add_scalar("EvalTestEpLen", sum(tlog_ep_len)/len(tlog_ep_len),eph)
            return sum(tlog_ep_ret)/len(tlog_ep_ret)


    #==================== Action module ====================
    def get_action(o, noise_scale=0.0):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * RD.randn(a.shape[0])
        return np.clip(a, env.act_ll, env.act_ul)

    #==================== Main Training Loop ====================
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    best_ep_ret = -9999

    o, ep_ret, ep_len = env.reset(), 0.0, 0.0
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # env.render()
        if (t+1) >= start_use_network and RD.rand() > 0.2:
            a = get_action(o, act_noise)
        else:
            a = env.random_action()

        # Step the env
        o2, r, d, info = env.step(a)
        ep_ret += r
        ep_len += 1

        if d:
            print("(Ep.Done)[Step_num] {:4} [Ep_ret] {:7.3f}  [Ep_len] {:4}  [INFO] S:{:4} | C:{:4} | TO:{:4}".format(
                env.step_num, ep_ret, ep_len, info['is_suc'], info['is_col'], info['is_timeout']
            ))

        d = False if info['is_timeout'] else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # make sure to update most recent observation!
        o = o2

        # End of trajectory handling
        if info['is_timeout'] or info['is_suc'] or info['is_col']:
            log_ep_ret.append(ep_ret)
            log_ep_len.append(ep_len)
            o, ep_ret, ep_len = env.reset(), 0.0, 0.0

        # Update handling
        if (t+1) % update_every == 0 and ((t+1) >= start_update_q or (t+1) >= start_update_pi):
            print("\n(UH)[t+1:{:6}], Start update. ep_ret: {:7.3f} | ep_len {:4} | step_num ({:3}/{:3})\n".format(
                t+1, ep_ret, ep_len, env.step_num, env.max_episode_steps
            ))
            if BN: # active estimate running mean and var continuously in BN
                ac.train()
            ac.to(device)
            ac_targ.to(device)
            if (t+1) >= start_update_pi:
                for _ in range(update_steps):
                    batch = replay_buffer.sample_batch(batch_size)
                    update_Q(data=batch)
                    update_PI(data=batch)
                    update_target_network()
                    pi_scheduler.step()
                    q_scheduler.step()
            else:
                for _ in range(int(update_steps/5)):
                    batch = replay_buffer.sample_batch(batch_size)
                    update_Q(data=batch)
                    update_target_network()
                    q_scheduler.step()
            if BN:
                ac.eval()

            ac.to(torch.device('cpu'))
            ac_targ.to(torch.device('cpu'))

            TBoard.add_scalar("loss_q",    sum(log_loss_q)/len(log_loss_q)      ,t+1)
            TBoard.add_scalar("q_vals",    sum(log_q_vals)/len(log_q_vals)      ,t+1)
            TBoard.add_scalar("q_target",  sum(log_q_target)/len(log_q_target)  ,t+1)
            TBoard.add_scalar("q_reward",  sum(log_q_reward)/len(log_q_reward)  ,t+1)
            TBoard.add_scalar("q2",        sum(log_q2)/len(log_q2)              ,t+1)
            if (t+1) >= start_update_pi:
                TBoard.add_scalar("loss_pi",   sum(log_loss_pi)/len(log_loss_pi)    ,t+1)
                print("LossQ:{:4.3f} LossPi:{:4.3f} Qval:{:4.3f} Qtarget:{:4.3f} Qr:{:4.3f} Q2:{:4.3f}\n".format(
                    sum(log_loss_q)/len(log_loss_q), sum(log_loss_pi)/len(log_loss_pi), sum(log_q_vals)/len(log_q_vals),
                    sum(log_q_target)/len(log_q_target), sum(log_q_reward)/len(log_q_reward), sum(log_q2)/len(log_q2))
                )
            log_loss_q, log_loss_pi, log_q_vals, log_q_target, log_q_reward, log_q2 = [], [], [], [], [], []


        # End of epoch handling
        if (t+1) >= start_update_pi and (t+1) % steps_per_epoch == 0:
            print("\n(EP)[t+1:{:6}], Epoch Logging. ep_ret: {:7.3f} | ep_len {:4} | step_num ({:3}/{:3})\n".format(
                t+1, ep_ret, ep_len, env.step_num, env.max_episode_steps
            ))
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            # test_agent(mode='train', epoch=epoch)
            test_ep_ret = test_agent(mode='test', eph=epoch)

            # Log info about epoch
            TBoard.add_scalar("Epoch",              epoch   ,epoch)
            TBoard.add_scalar("TotalEnvInteracts",  t       ,epoch)
            TBoard.add_scalar("EpRet",  sum(log_ep_ret)/len(log_ep_ret)  ,epoch)
            TBoard.add_scalar("EpLen",  sum(log_ep_len)/len(log_ep_len)  ,epoch)
            log_ep_ret, log_ep_len = [], []

            elapsed_time = time.time()-start_time
            TBoard.add_scalar("Time_sec",   elapsed_time    ,epoch)
            TBoard.add_scalar("Time_min",   elapsed_time/60 ,epoch)

            # Save model
            if test_ep_ret >= best_ep_ret :
                best_ep_ret = test_ep_ret
                torch.save(ac.state_dict(), log_dir + '/best_model.pt')
                print("================ Saved best_model.pt (ep: {}) ================".format(epoch))
