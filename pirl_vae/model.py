from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        residul = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)

        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        x = x + residul
        return x

class VanillaVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims):
        super(VanillaVAE, self).__init__()
        self.im_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = [in_channels] + hidden_dims

        modules = []

        # Build Encoder
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv3d(self.hidden_dims[i], out_channels=self.hidden_dims[i+1], kernel_size=(3, 5, 3), stride=(2, 2, 2), padding=(1, 2, 1)),
                    BasicBlock(self.hidden_dims[i+1], self.hidden_dims[i+1])
                )
            )
        modules.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*modules)
        # from torchsummary import summary
        # print(summary(self.encoder, (1, 64, 64, 64), device='cpu'))

        flatten_dim = self.hidden_dims[-1]*(2*2*2)
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_var = nn.Linear(flatten_dim, latent_dim)

        self.hidden_dims.reverse()
        modules = []
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, flatten_dim)
        for i in range(len(self.hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Conv3d(self.hidden_dims[i],
                              self.hidden_dims[i + 1],
                              kernel_size=(1, 1, 1)),
                    nn.Upsample(scale_factor=2),
                    BasicBlock(self.hidden_dims[i + 1], self.hidden_dims[i + 1])
                )
            )
        modules.append(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv3d(self.hidden_dims[-2], self.hidden_dims[-1], (1, 1, 1)),
                nn.Upsample(scale_factor=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)
        # print(self.decoder)
        # print(self.hidden_dims)
        # from torchsummary import summary
        # print(summary(self.decoder, (512, 2, 2, 2), device='cpu'))


    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], 2, 2, 2)
        x = self.decoder(x)
        return x

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, kld_weight):
        occ_ele = input.sum()
        total_ele = input.numel()
        loss_weight = input * (1 - occ_ele/total_ele) + -1 * (input - 1) * (occ_ele/total_ele)
        recons_loss = F.binary_cross_entropy(recons, input, loss_weight)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples,
               current_device):
        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def reconstruction(self, x):
        return self.forward(x)[0]

if __name__ == '__main__':
    model = VanillaVAE(1, 64, hidden_dims=[32, 64, 128, 256, 512])
    print(model)
    model = model.cuda()
    print("\n================ Model summary ================")
    from torchsummary import summary
    print(summary(model, (1, 64, 64, 64), device='cuda'))

    print("\n================ Output size check ================")
    x = torch.randn(16, 1, 64, 64, 64).cuda()
    y = model(x)
    print("Model Output size:", y[0].size())

    print("\n================ loss check ================")
    x = torch.randn(16, 1, 64, 64, 64).cuda()
    result = model(x)
    loss = model.loss_function(*result, kld_weight=0.005)
    print(loss)