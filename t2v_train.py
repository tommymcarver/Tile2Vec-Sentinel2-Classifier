
import os
import torch
from torch import optim
from time import time
import numpy as np

from tileloader import tileInfo, triplet_dataloader
from tileloader import tileInfo, getTriplet, TileTripletsDataset, triplet_dataloader
from tilenetDownsized import make_tilenet, TileNet
from src.training import train_triplet_epoch
import random
import multiprocessing

import argparse

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

device = args.device

print("Using {} cpu cores".format(multiprocessing.cpu_count()))
torch.set_num_threads(multiprocessing.cpu_count())

# Number of bands/input channels to the neural net
in_channels = 4

# Dimensions to feed to the growing size of the layers in the neural net
z_dim = 64


myTileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
myTileNet.train()
myTileNet = myTileNet.to(device=device)
print('TileNet set up complete.')


data_dir = '../../../data'

# Which type of data to train on
selection = 'corn_belt'

img_dir = os.path.join(data_dir, selection)

# Grabbing all of the downloaded image names into an array
img_list = []
for root, dirs, files in os.walk(img_dir):
    img_list = files
    break

# Setting default dataloading and dataset parameters for setting training up
bands = in_channels
batch_size =100
tiles_per_image = 48000
shuffle = False
num_workers = 3
seed = False
seed_num = 100

tileInformation = tileInfo(tile_size=50, neighborhood=100, val_type='float32')

# Determining if a preloaded seed should be
if seed:
    print("{} is your seed for this training".format(seed_num))
    np.random.seed(seed_num)
    dataloader = triplet_dataloader(img_list, tileInformation, tiles_per_image=tiles_per_image, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
else:
    rand = random.randint(1,10000)
    np.random.seed(rand)
    print(str(rand) + " is your seed for this training")
    dataloader = triplet_dataloader(img_list, tileInformation, tiles_per_image=tiles_per_image, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

print("Dataloader set up complete")

# Setting up the optimizer
lr = 1e-3
optimizer = optim.Adam(myTileNet.parameters(), lr=lr, betas=(0.5, 0.999))

# Setting up the actual training parameters for each epoch
epochs = 1
margin = 10
l2 = 0.01
print_every =1000
save_models = True

model_dir = 'models' #path to where you want to save the model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Training
t0 = time()
results_fn = os.path.join(model_dir, 'fileTest.ckpt')
with open(results_fn, 'w') as file:
    print('Begin training........')
    for epoch in range(0, epochs):
        (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
            myTileNet,  dataloader, optimizer, epoch+1, device=device, margin=margin, l2=l2,
            print_every=print_every, t0=t0)

import pickle
#save model after finished
model_fn = os.path.join(model_dir, 'all_of_sent')#put the name you  want to give the model here
torch.save(myTileNet.state_dict(), model_fn)
