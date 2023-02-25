import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataloader import get_dataloader
from model import VanillaVAE


def main(args):
    TBoard = SummaryWriter(args.log_path)

    trn_dataset, eval_dataset, trn_loader, eval_loader = get_dataloader(batch_size=args.bs, num_workers=4)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = VanillaVAE( in_channels=1,
                        latent_dim=args.z_dim, 
                        hidden_dims=[32, 64, 128, 256, 512]).to(device)
    if args.cuda:
        model.cuda()


    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.WD)
    scheduler = optim.lr_scheduler.StepLR(  optimizer=optimizer,
                                            step_size=int(args.ep/args.steps),
                                            gamma=0.7) # 0.7^5 = 0.16

    Best_loss = np.inf
    n_trn_batch = len(trn_loader)
    n_eval_batch = len(eval_loader)
    print(f"[INFO] n_trn_batch: {n_trn_batch}, n_eval_batch: {n_eval_batch} | bs: {args.bs}")
    for epoch in range(args.ep): 
        ep_start_time = time.time()
        ##################### Train #####################
        model.train()
        AVG_loss = 0
        BCE_loss = 0
        KLD_loss = 0 
        wKLD = min( max(epoch-args.start_wKLD,0)/args.duration_wKLD, 1)
        for i, mini_batch in enumerate(trn_loader):
            print(f"[{i}/{n_trn_batch}]", end='\r')
            input_data = mini_batch.to(device)

            optimizer.zero_grad()

            result = model(input_data)

            loss_dict = model.loss_function(*result, kld_weight=args.wKLDMAX*wKLD)
            loss = loss_dict['loss']

            loss.backward()
            optimizer.step()
            
            AVG_loss += loss.item()
            BCE_loss += loss_dict['Reconstruction_Loss'].item()
            KLD_loss += loss_dict['KLD'].item()

        scheduler.step()
        print("=============================================================================================")
        print(f"[Trn] EP:{epoch:5}| AVG_L:{AVG_loss/n_trn_batch:.4f}| BCE_L:{BCE_loss/n_trn_batch:.4f}, KLD_L:{KLD_loss/n_trn_batch:.4f}")

        TBoard.add_scalar('Trn_AVG_L', AVG_loss/n_trn_batch, epoch)
        TBoard.add_scalar('Trn_BCE_L', BCE_loss/n_trn_batch, epoch)
        TBoard.add_scalar('Trn_KLD_L', KLD_loss/n_trn_batch, epoch)
        TBoard.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        TBoard.add_scalar('wKLD', args.wKLDMAX*wKLD, epoch)

        model.eval()

        ##################### Evaluation #####################
        model.eval()
        Eval_AVG_loss = 0
        Eval_BCE_loss = 0
        Eval_KLD_loss = 0 
        with torch.no_grad():
            for i, mini_batch in enumerate(eval_loader):
                print(f"[{i}/{n_eval_batch}]", end='\r')
                input_data = mini_batch.to(device)

                result = model(input_data)

                loss_dict = model.loss_function(*result, kld_weight=args.wKLDMAX*wKLD)
                loss = loss_dict['loss']
            
                Eval_AVG_loss += loss.item()
                Eval_BCE_loss += loss_dict['Reconstruction_Loss'].item()
                Eval_KLD_loss += loss_dict['KLD'].item()

        print(f"[Eval]EP:{epoch:5}| AVG_L:{Eval_AVG_loss/n_eval_batch:.4f}| BCE_L:{Eval_BCE_loss/n_eval_batch:.4f}, KLD_L:{Eval_KLD_loss/n_eval_batch:.4f}")

        TBoard.add_scalar('Eval_AVG_L', Eval_AVG_loss/n_eval_batch, epoch)
        TBoard.add_scalar('Eval_BCE_L', Eval_BCE_loss/n_eval_batch, epoch)
        TBoard.add_scalar('Eval_KLD_L', Eval_KLD_loss/n_eval_batch, epoch)

        if Best_loss > Eval_AVG_loss/n_eval_batch:
            Best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(args.log_path,'VAE_best.pkl'), _use_new_zipfile_serialization=False)
        torch.save(model.state_dict(), os.path.join(args.log_path,'VAE_last.pkl'), _use_new_zipfile_serialization=False)

        ep_elapsed_time = time.time() - ep_start_time
        print(f"ep_elapsed_time: {ep_elapsed_time:.2f}")
    ### Save & close
    TBoard.close()

"""
CUDA_VISIBLE_DEVICES=1 python3 train.py --ep 2000 --bs 512 --lr 3e-4 --steps 10 --z_dim 128 --wKLDMAX 0.00001 --start_wKLD 0 --duration_wKLD 2000 --cuda
CUDA_VISIBLE_DEVICES=0 python3 train.py --ep 300 --bs 256 --lr 3e-4 --steps 3 --z_dim 64 --wKLDMAX 1e-10 --start_wKLD 1e-20 --duration_wKLD 100 --cuda
python3 train.py --ep 2000 --bs 32 --lr 4e-5 --steps 20 --z_dim 64 --wKLDMAX 0.00001 --start_wKLD 0.00000001 --duration_wKLD 2000 --cuda
"""
print('[start script] {}'.format(os.path.abspath(__file__)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ep', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--WD', type=float, default=0.0)
    parser.add_argument('--z_dim', type=int, default=24)
    parser.add_argument('--wKLDMAX', type=float, default=0)
    parser.add_argument('--start_wKLD', type=float, default=0)
    parser.add_argument('--duration_wKLD', type=float, default=10)
    parser.add_argument('--cuda', action='store_true')

    parser.add_argument('--memo', type=str, default='')

    args = parser.parse_args()
    print(args)

    log_dir = args.log_path + '/' + ("{}_".format(args.memo) if args.memo else "") \
              + "ep_{}_steps{}_lr_{}_bs_{}_z_dim{}_WD_{}_wKLD_MAX_{}_start{}_duration{}_cuda{}".format(
                                                                args.ep , 
                                                                args.steps,
                                                                args.lr ,
                                                                args.bs , 
                                                                args.z_dim,
                                                                args.WD,
                                                                args.wKLDMAX,
                                                                args.start_wKLD,
                                                                args.duration_wKLD,
                                                                args.cuda
                                                                )
    args.log_path = log_dir
    os.makedirs(args.log_path, exist_ok=True)
    main(args)
