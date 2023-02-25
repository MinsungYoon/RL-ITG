import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn

from dataloader import get_dataloader
from model import VanillaVAE

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import functional as F
import matplotlib.pyplot as plt



def main(model_path, model_name, use_cuda):

    log_path = os.path.join("./result", model_path.split('/')[1])
    os.makedirs(log_path, exist_ok=True)

    log_train_path = os.path.join(log_path, "train")
    log_eval_path = os.path.join(log_path, "eval")
    os.makedirs(log_train_path, exist_ok=True)
    os.makedirs(log_eval_path, exist_ok=True)

    trn_dataset, eval_dataset, trn_loader, eval_loader = get_dataloader(batch_size=1, num_workers=4)

    z_dim = int( model_path[ model_path.find('z_dim')+len('z_dim') : model_path.find('_WD_')] )
    model = VanillaVAE( in_channels=1,
                        latent_dim=z_dim,
                        hidden_dims=[32, 64, 128, 256, 512])
    state_dict = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(state_dict)

    print(model)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
    else:
        device = torch.device('cpu')

    def calculate_metrics(mode, loader, device):
        total_mse = 0
        latent_features = []
        for i, mini_batch in enumerate(loader):
            input_data = mini_batch.to(device)
            result = model(input_data)

            latent_features.append(result[2].cpu().detach().squeeze(0).numpy())

            MSE = F.mse_loss(result[1], result[0]).cpu()
            total_mse += MSE.item()

            print("{}-{}: {}, {}".format(mode, i, result[0].max().item(), result[0].min().item()))

            def save_fig(original, recon, m, ii):
                fig = plt.figure(figsize=(20, 15))

                ax = fig.add_subplot(121, projection='3d')
                data = (original.cpu() > 0.5).float().detach().squeeze(0).squeeze(0).numpy()
                x, y, z = data.nonzero()
                ax.scatter(x, y, z, c=z, alpha=1)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.set_xlim([0, 64])
                ax.set_ylim([0, 64])
                ax.set_zlim([0, 64])

                ax2 = fig.add_subplot(122, projection='3d')
                data = (recon.cpu() > 0.5).float().detach().squeeze(0).squeeze(0).numpy()
                x, y, z = data.nonzero()
                ax2.scatter(x, y, z, c=z, alpha=1)
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax2.set_zlabel('z')
                ax2.set_xlim([0, 64])
                ax2.set_ylim([0, 64])
                ax2.set_zlim([0, 64])

                if m == "train":
                    plt.savefig("{}/{}.png".format(log_train_path, ii))
                else:
                    plt.savefig("{}/{}.png".format(log_eval_path, ii))
                plt.close(fig)
            if mode == "train":
                if i % 100 == 0:
                    save_fig(result[1], result[0], mode, i)
            else:
                if i % 100 == 0:
                    save_fig(result[1], result[0], mode, i)
        latent_features = np.array(latent_features)

        pca_model = PCA(n_components=2)
        pca_features = pca_model.fit_transform(latent_features)

        tsne = TSNE(random_state=0)
        tsne_features = tsne.fit_transform(latent_features)

        return total_mse/len(loader), pca_features, tsne_features

    trn_mse, trn_pca_features, trn_tsne_features = calculate_metrics('train', trn_loader, device)
    print(f"Trn  - Avg_MSE: {trn_mse}")
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(trn_pca_features[:,0], trn_pca_features[:,1])
    ax[1].scatter(trn_tsne_features[:,0], trn_tsne_features[:,1])
    fig.suptitle('Train set latent feature plot')
    ax[0].set_title("PCA")
    ax[1].set_title("T-sne")
    plt.savefig("{}/TrainSet_Embedding.png".format(log_train_path))

    eval_mse, eval_pca_features, eval_tsne_features = calculate_metrics('eval', eval_loader, device)
    print(f"Eval - Avg_MSE: {eval_mse}")
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(eval_pca_features[:,0], eval_pca_features[:,1])
    ax[1].scatter(eval_tsne_features[:,0], eval_tsne_features[:,1])
    fig.suptitle('Evaluation set latent feature plot')
    ax[0].set_title("PCA")
    ax[1].set_title("T-sne")
    plt.savefig("{}/TestSet_Embedding.png".format(log_eval_path))


print('[start script] {}'.format(os.path.abspath(__file__)))
if __name__ == '__main__':

    model_path = 'logs/Final_ep_200_steps4_lr_0.0001_bs_128_z_dim64_WD_0.0_wKLD_MAX_1e-06_start0.0_duration100.0_cudaTrue'
    model_name = 'VAE_last.pkl'
    use_cuda = False

    main(model_path, model_name, use_cuda)
