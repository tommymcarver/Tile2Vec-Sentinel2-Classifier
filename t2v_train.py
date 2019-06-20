import sys
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

cuda = torch.cuda.is_available()

in_channels = 4
z_dim = 64

myTileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
myTileNet.train()
if cuda: myTileNet.cuda()
print('TileNet set up complete.')


data_dir = '../../../data'

# Which type of data to train on
selection = 'corn_belt'

img_dir = os.path.join(data_dir, selection)

img_list = []
for root, dirs, files in os.walk(img_dir):
    img_list = dirs
    break

bands = in_channels
batch_size = 1250
shuffle = False
num_workers = 4
seed = True

tileInformation = tileInfo(tile_size=50, neighborhood=100, val_type='float32')

if seed:
    seed = 100
    np.random.seed(seed)
    dataloader = triplet_dataloader(img_list, tileInformation, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
else:
    rand = random.randint(1,10000)
    np.random.seed(rand)
    print(str(torch.initial_seed()) + " is your seed for this training")
    dataloader = triplet_dataloader(img_list, tileInformation, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

print("Dataloader set up complete")

lr = 1e-3
optimizer = optim.Adam(myTileNet.parameters(), lr=lr, betas=(0.5, 0.999))

epochs = 1
margin = 10
l2 = 0
print_every = 1000
save_models = True

model_dir = 'models' #path to where you want to save the model
if not os.path.exists(model_dir): os.makedirs(model_dir)

t0 = time()
results_fn = os.path.join(model_dir, 'fileTest.ckpt')
with open(results_fn, 'w') as file:
    print('Begin training........')
    for epoch in range(0, epochs):
        (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
            myTileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
            print_every=print_every, t0=t0)

import pickle
#save model after finished
model_fn = os.path.join(model_dir, 'tile2vec_model')#put the name you  want to give the model here
torch.save(myTileNet.state_dict(), model_fn)
