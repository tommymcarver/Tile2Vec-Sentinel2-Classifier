from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import os
import torch

from sklearn.preprocessing import LabelEncoder
from fix_labels import crop_notcrop_test, crop_notcrop_train

class Tiles(Dataset):
    def __init__(self, model, tile_dir, n_tiles, test=False, transform=None, n_trials=1000):
        self.transform = transform
        self.labels = np.load(os.path.join(tile_dir, 'y.npy'))
        if not(test):
            self.labels = crop_notcrop_train(n_tiles, self.labels)
        else:
            self.labels = crop_notcrop_test(n_tiles, self.labels)
        if not(test):
            self.tiles = get_train_tiles(n_tiles, model, tile_dir)
        else:
            self.tiles = get_test_tiles(n_tiles, model, tile_dir)

    def __len__(self):
        return np.ma.size(self.tiles, axis=0)

    def __getitem__(self, idx):
        return (self.tiles[idx], self.labels[idx])


def get_train_tiles(n_tiles, model, tile_dir, z_dim=64):
    cuda = torch.cuda.is_available()
    X = np.zeros((n_tiles, 4, 50, 50))
    for idx in range(n_tiles):
        tile = np.load(os.path.join(tile_dir, '{}tile.npy'.format(idx + 1))).astype('double')
        # Get first 4 NAIP channels (5th is CDL mask)
        tile = tile[:, :, :4]
        # Rearrange to PyTorch order
        tile = np.moveaxis(tile, -1, 0)
        tile = np.expand_dims(tile, axis=0)
        tile = tile/255
        tile = torch.from_numpy(tile).float()
        tile = Variable(tile)
        X[idx,:] = tile
    return X

def get_test_tiles(n_tiles, model, tile_dir, z_dim=64):
    cuda = torch.cuda.is_available()
    X = np.zeros((n_tiles, 4, 50, 50))
    count = 0
    for idx in range(1000 -n_tiles, 1000):
        tile = np.load(os.path.join(tile_dir, '{}tile.npy'.format(idx + 1))).astype('double')
        # Get first 4 NAIP channels (5th is CDL mask)
        tile = tile[:, :, :4]
        # Rearrange to PyTorch order
        tile = np.moveaxis(tile, -1, 0)
        tile = np.expand_dims(tile, axis=0)
        tile = tile/255
        tile = torch.from_numpy(tile).float()
        tile = Variable(tile)
        X[count,:] = tile
        count = count + 1
    return X

def tile_dataloader(model, tile_dir, n_tiles, test=False, num_workers=4, batch_size=4, shuffle=True):
    dataset = Tiles(model, tile_dir, n_tiles, test=test)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader
