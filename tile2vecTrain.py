#!/usr/bin/env python
# coding: utf-8

# ## Tile2Vec Streaming Training
#
# Uses provided image files to train a net, "streaming" the training examples as necessary during the training

# In[1]:


import sys
import os
import torch
from torch import optim
from time import time
import numpy as np


# In[2]:


#tile2vec_dir = ''#path to your Tile2Vec folder
#sys.path.append(tile2vec_dir)


# In[3]:


from streamingDataloader import tileInfo, makeImageDict, getTriplet, TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader, makeID
from tilenetDownsized import make_tilenet, TileNet
from src.training import prep_triplets, train_triplet_epoch


# In[ ]:


# CUDA Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()


# In[ ]:


# Fetch tiles, set basic parameters

img_type = 'landsat'
img_dir = '../../../data/images' #path to file containing your images
bands = 4
augment = True
batch_size = 100
shuffle = True
num_workers = 0
n_triplets = 100000


# In[ ]:
tileInformation= tileInfo(tile_size=50,neighborhood=100,val_type='float32')


# In[ ]:
#
# Dictionary of images keyed by name of image file, stored into padded arrays with according tile info

imageInput = makeID(img_dir,["14SLG"], tileInformation)


#imageInput = makeID(img_dir,["14TPP", "14TPN.tif", "16TCM", "14TMN.tif"], tileInformation)



#imageInput= makeImageDict(img_dir,["train1.png"],tileInformation)#put name(s) of the directories you want to use in the brackets


# In[ ]:


dataloader = triplet_dataloader(imageInput, tileInformation, img_type, bands=bands, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                n_triplets=n_triplets, pairs_only=True)
print('Dataloader set up complete.')


# In[ ]:


in_channels = bands
z_dim = 64


# In[ ]:


myTileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
myTileNet.train()
if cuda:
    myTileNet.cuda()
print('TileNet set up complete.')


# In[ ]:

# Can change these values to find the optimal optimizer
lr = 1e-3
optimizer = optim.Adam(myTileNet.parameters(), lr=lr, betas=(0.5, 0.999))


# In[ ]:


epochs = 5
margin = 10
l2 = 0
print_every = 500
save_models = True


# In[ ]:


model_dir = 'models' #path to where you want to save the model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# In[ ]:


t0 = time()
results_fn = os.path.join(model_dir, 'fileTest.ckpt')
with open(results_fn, 'w') as file:
    print('Begin training........')
    for epoch in range(0, epochs):
        (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
            myTileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
            print_every=print_every, t0=t0)


# In[ ]:


import pickle
#save model after finished
model_fn = os.path.join(model_dir, 'tile2vec_model')#put the name you  want to give the model here
torch.save(myTileNet.state_dict(), model_fn)
