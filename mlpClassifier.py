import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
from src.training import prep_tile
from classifierLoading import tile_dataloader
from net import ResidualBlock, Net
import multiprocessing
from torch import optim
from time import time

import argparse

cuda = torch.cuda.is_available()
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
print(device)

print("Using {} cpu cores".format(multiprocessing.cpu_count()))
torch.set_num_threads(multiprocessing.cpu_count())


# Setting up the model
z_dim = 64
num_blocks = [2, 2, 2, 2, 2]
in_channels = 4
model = Net(in_channels=in_channels, num_blocks=num_blocks, z_dim=z_dim)
model_dict = model.state_dict()
checkpoint = torch.load(os.path.join(
    "models", "all_of_sent"))
#checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
model_dict.update(checkpoint)
model.load_state_dict(model_dict)
for param in model.parameters():
    param.requires_grad = False
for idx, child in enumerate(model.children()):
    if idx < 2 and idx > 6:
        for param in child.parameters():
            param.requires_grad = True
model = model.to(device=device)
model = model.train()
print("Model successfully loaded")


img_list = []
for root, dirs, files in os.walk('../../../data/corn_belt'):
    img_list = files
    break
batch_size=100
dataloader = tile_dataloader(img_list, batch_size)

lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(.9,.999))

epochs=2
save_models=True
model_dir = 'classifier_models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
t0 = time()
print('Begin Classifier Training......')
for epoch in range(0, epochs):
    running_loss = 0.0
    for idx, data in enumerate(dataloader):
        tile, label = data
        tile, label = prep_tile(tile, label, device=device)
        optimizer.zero_grad()
        print(tile)
        outputs = model(tile.float().cuda())
        loss = model.loss()
        loss = loss(outputs, label.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("Batch Loaded")
        if idx % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx + 1, running_loss / 5))
            running_loss = 0.0
        

