import numpy as np
import time
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tensorboardX import SummaryWriter

import core

# import argparse
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--algo', type=str, default='')
#     parser.add_argument('--memo', type=str, default='')
#     arg_parser = parser.parse_args()

demo_info = {
    "path": "/home/minsungyoon/nvme/pirl_ws/src/pirl/script/imitation_learning/",
    "model": "MANN",
    "data_wps":  [4, 5, 6, 7],
    "ep":   10000,
    "bs":   150,
    "step": 500,
    "lr":   3e-4,
    "wd":   1e-5,
    "grad_clip": 0.0,
    "hids": [],
    "BN": False,
    "DR": False,
    "Actll": "vel"
}

exp_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
    demo_info["model"],
    "ep", demo_info["ep"],
    "lr", demo_info["lr"],
    "bs", demo_info["bs"],
    "step", demo_info["step"],
    "wd", demo_info["wd"],
    "gclip", demo_info["grad_clip"],
    "hids", demo_info["hids"],
    "BN", demo_info["BN"],
    "DR", demo_info["DR"],
    "Actll", demo_info["Actll"]
)

memo = ""
if memo:
    exp_name = memo + "__" + exp_name
log_dir = "./log/" + exp_name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

BC_model_path = log_dir + '/bc_model.pt'
BC_best_model_path = log_dir + '/bc_best_model.pt'

TBoard = SummaryWriter(log_dir)

loaded_data_x = []
loaded_data_y = []
for w in demo_info["data_wps"]:
    path = demo_info["path"]+str(w)+"_wps_data_train.npz"
    loaded_data_x.append(np.load(path)['x'])
    loaded_data_y.append(np.load(path)['y'])
BC_DATA_X_TRAIN = np.concatenate(loaded_data_x, axis=0).astype(np.float32)
BC_DATA_Y_TRAIN = np.concatenate(loaded_data_y, axis=0).astype(np.float32)

loaded_data_x = []
loaded_data_y = []
for w in demo_info["data_wps"]:
    path = demo_info["path"]+str(w)+"_wps_data_test.npz"
    loaded_data_x.append(np.load(path)['x'])
    loaded_data_y.append(np.load(path)['y'])
BC_DATA_X_TEST = np.concatenate(loaded_data_x, axis=0).astype(np.float32)
BC_DATA_Y_TEST = np.concatenate(loaded_data_y, axis=0).astype(np.float32)

BC_DATA_X_TRAIN = torch.from_numpy(BC_DATA_X_TRAIN)
BC_DATA_Y_TRAIN = torch.from_numpy(BC_DATA_Y_TRAIN)
BC_DATA_X_TEST = torch.from_numpy(BC_DATA_X_TEST)
BC_DATA_Y_TEST = torch.from_numpy(BC_DATA_Y_TEST)

###################################################################################### model
obs_dim = BC_DATA_X_TRAIN[0].shape[0]
act_dim = BC_DATA_Y_TRAIN[0].shape[0]

if demo_info["Actll"] == "vel":
    act_ll = np.array([-1.256, -1.454, -1.571, -1.521, -1.571, -2.268, -2.268])
    act_ul = np.array([1.256, 1.454, 1.571, 1.521, 1.571, 2.268, 2.268])
else:
    act_ll = np.array([-6.28]*7)
    act_ul = np.array([6.28]*7)
act_mean = (act_ul + act_ll)/2
act_std = (act_ul - act_ll)/2
BN = demo_info["BN"]
DR = demo_info["DR"]
hidden_sizes = demo_info["hids"]
activation = nn.ELU

if demo_info["model"] == "MLP":
    pi = core.NormalMLP(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)
    with_log_mu_std = False
elif demo_info["model"] == "MANN":
    index_gating = np.arange(7,72) # cur links' position and rotation
    # index_gating = np.arange(72,133) # local future target relative to current EE pose.
    pi = core.MANN(index_gating, n_expert_weights=8, hg=[512, 256],
                   n_input_motion=133, n_output_motion=7, h=[1024, 512],
                   drop_prob_gat=0.0, drop_prob_mot=0.0)
    with_log_mu_std = False
elif demo_info["model"] == "SGMLPv2":
    pi = core.SquashedGaussianMLPActor_ver2(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)
    with_log_mu_std = True
else:
    pi = core.SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_mean, act_std, BN, DR)
    with_log_mu_std = True

def count_vars(model):
    print("[Model Param] #: {}".format(sum([np.prod(p.shape) for p in model.parameters()])))
count_vars(pi)
print(pi)
print("[INFO] Dim obs: {}, act: {}".format(obs_dim, act_dim))

###################################################################################### main

BC_trn_loader = DataLoader(TensorDataset(BC_DATA_X_TRAIN, BC_DATA_Y_TRAIN),
                           batch_size=  demo_info["bs"],
                           shuffle=     True,
                           num_workers= 4,
                           drop_last=   True)
BC_test_loader = DataLoader(TensorDataset(BC_DATA_X_TEST, BC_DATA_Y_TEST),
                           batch_size=  demo_info["bs"],
                           shuffle=     False,
                           num_workers= 4,
                           drop_last=   True)

BC_optimizer = optim.Adam( filter(lambda p: p.requires_grad, pi.parameters()), lr=demo_info["lr"], weight_decay=demo_info["wd"])
BC_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(BC_optimizer,
                                                              T_0=demo_info["step"],
                                                              T_mult=1,
                                                              eta_min=float(demo_info["lr"]*0.001),
                                                              last_epoch=-1)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    exit()

pi = pi.to(device)
n_trn_batch = len(BC_trn_loader)
n_test_batch = len(BC_test_loader)
print("[BC] Total Train Data: {}, # of trn batch iter: {} | bs: {}".format(BC_DATA_X_TRAIN.shape[0], n_trn_batch, demo_info["bs"]))
print("[BC] Total Test Data: {}, # of test batch iter: {} | bs: {}".format(BC_DATA_X_TEST.shape[0], n_test_batch, demo_info["bs"]))
Best_loss = np.inf
for epoch in range(demo_info["ep"]):
    pi.train()
    log_BC_logp_pi, log_BC_logp_pi_min, log_BC_logp_pi_max = [], [], []
    log_BC_mu, log_BC_mu_min, log_BC_mu_max = [], [], []
    log_BC_std, log_BC_std_min, log_BC_std_max = [], [], []
    ##################### Train #####################
    MSE_loss = 0
    for i, (BX, BY) in enumerate(BC_trn_loader):
        print("\r[{}/{}]".format(i, n_trn_batch)),
        BX = BX.to(device)
        BY = BY.to(device)

        BC_optimizer.zero_grad()
        if with_log_mu_std:
            action, logp_pi, mu, std = pi(BX)
        else:
            action = pi(BX)
        loss = F.mse_loss(action, BY)
        loss.backward()
        if demo_info["grad_clip"] > 0.0:
            torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=demo_info["grad_clip"])
        BC_optimizer.step()

        MSE_loss += loss.item()
        if with_log_mu_std:
            log_BC_logp_pi.append(logp_pi.cpu().mean())
            log_BC_logp_pi_min.append(logp_pi.cpu().min())
            log_BC_logp_pi_max.append(logp_pi.cpu().max())
            log_BC_mu.append(mu.cpu().mean())
            log_BC_mu_min.append(mu.cpu().min())
            log_BC_mu_max.append(mu.cpu().max())
            log_BC_std.append(std.cpu().mean())
            log_BC_std_min.append(std.cpu().min())
            log_BC_std_max.append(std.cpu().max())

    BC_scheduler.step()
    print("=============================================================================================")
    print("[BC] EP:{:5}| MSE_L:{:.4f}".format(
        epoch, MSE_loss/n_trn_batch
    ))
    if with_log_mu_std:
        TBoard.add_scalar('BC_logp_pi',     sum(log_BC_logp_pi)/len(log_BC_logp_pi), epoch)
        TBoard.add_scalar('BC_logp_pi_min', sum(log_BC_logp_pi_min)/len(log_BC_logp_pi_min), epoch)
        TBoard.add_scalar('BC_logp_pi_max', sum(log_BC_logp_pi_max)/len(log_BC_logp_pi_max), epoch)
        TBoard.add_scalar('BC_mu',      sum(log_BC_mu)/len(log_BC_mu), epoch)
        TBoard.add_scalar('BC_mu_min',  sum(log_BC_mu_min)/len(log_BC_mu_min), epoch)
        TBoard.add_scalar('BC_mu_max',  sum(log_BC_mu_max)/len(log_BC_mu_max), epoch)
        TBoard.add_scalar('BC_std',     sum(log_BC_std)/len(log_BC_std), epoch)
        TBoard.add_scalar('BC_std_min', sum(log_BC_std_min)/len(log_BC_std_min), epoch)
        TBoard.add_scalar('BC_std_max', sum(log_BC_std_max)/len(log_BC_std_max), epoch)

    TBoard.add_scalar('BC_L', MSE_loss/n_trn_batch, epoch)
    TBoard.add_scalar('LR_BC', BC_optimizer.param_groups[0]['lr'], epoch)

    ##################### Evaluation #####################
    pi.eval()
    Eval_AVG_loss = 0
    with torch.no_grad():
        for i, (BX, BY) in enumerate(BC_test_loader):
            BX = BX.to(device)
            BY = BY.to(device)

            if with_log_mu_std:
                action, _, _, _ = pi(BX, deterministic=True, with_logprob=False)
            else:
                action = pi(BX)

            loss = F.mse_loss(action, BY)

            Eval_AVG_loss += loss.item()
    print("[BC] TEST MSE_L:{:.4f}".format(
        Eval_AVG_loss/n_test_batch
    ))
    TBoard.add_scalar('BC_L_TEST', Eval_AVG_loss/n_test_batch, epoch)

    if Best_loss > Eval_AVG_loss/n_test_batch:
        Best_loss = Eval_AVG_loss/n_test_batch
        torch.save(pi.state_dict(), BC_best_model_path)
    torch.save(pi.state_dict(), BC_model_path)

