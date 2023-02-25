import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TerrainDataset(Dataset):
    """TerrainDataset"""
    def __init__(self, data_dir, start_idx, end_idx, resize_dim):
        self.resize_dim = resize_dim
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.data_list.sort(key=lambda x: int(x[4:x.find(".npy")]))
        self.data_list = self.data_list[start_idx: end_idx]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        occ_name = self.data_dir + self.data_list[idx]
        occ = np.load(occ_name)
        if np.random.rand() > 0.5:
            occ = np.flip(occ, axis=1).copy() # flip doesn't really flip data just index.
        occ = torch.from_numpy(occ)
        occ = occ.to(torch.float32)
        occ = torch.nn.functional.interpolate(occ.unsqueeze(0).unsqueeze(0), (self.resize_dim, self.resize_dim, self.resize_dim)).squeeze(0)
        return occ

def get_dataloader(batch_size, num_workers):
    trn_dataset = TerrainDataset(
                    data_dir='/data/torm_data/obs/scene_occ/', start_idx=0, end_idx=4500, resize_dim=64
                )
    eval_dataset = TerrainDataset(
                    data_dir='/data/torm_data/obs/scene_occ/', start_idx=4500, end_idx=5000, resize_dim=64
                )
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return trn_dataset, eval_dataset, trn_loader, eval_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = TerrainDataset(
                    data_dir='/data/torm_data/obs/scene_occ/', start_idx=0, end_idx=4500, resize_dim=64
                )

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(121, projection='3d')
    data = dataset[1520].squeeze(0).numpy()
    print(data.shape)
    x, y, z = data.nonzero()
    ax.scatter(x, y, z, c=z, alpha=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([0, 64])
    ax.set_ylim([0, 64])
    ax.set_zlim([0, 64])

    ax2 = fig.add_subplot(122, projection='3d')
    data = dataset[1521].squeeze(0).numpy()
    print(data.shape)
    x, y, z = data.nonzero()
    ax2.scatter(x, y, z, c=z, alpha=1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim([0, 64])
    ax2.set_ylim([0, 64])
    ax2.set_zlim([0, 64])
    plt.savefig("test")
    plt.show()


    # for i in range(10):
        # print(dataset[i])

    # import ipdb
    # ipdb.set_trace()
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4, drop_last=True)
    # for i, mini_batch in enumerate(dataloader):
        # print(f"i: {i}, mini_batch.size(): {mini_batch.size()}")

    # import ipdb
    # ipdb.set_trace()

