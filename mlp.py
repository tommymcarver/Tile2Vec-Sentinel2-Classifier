import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time

from classifierLoading import tile_dataloader
from net import ResidualBlock, Net


classes = ('not crop', 'crop')
cuda = torch.cuda.is_available()
# Setting up the model
z_dim = 64
num_blocks = [2, 2, 2, 2, 2]
in_channels = 4
model = Net(in_channels=in_channels, num_blocks=num_blocks, z_dim=z_dim)
model_dict = model.state_dict()
checkpoint = torch.load(os.path.join(
    "models", "tile2vec_model"), map_location='cpu')
checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
model_dict.update(checkpoint)
model.load_state_dict(model_dict)
for param in model.parameters():
    param.requires_grad = False
for idx, child in enumerate(model.children()):
    if idx < 3 or idx > 6:
        for param in child.parameters():
            param.requires_grad = True
if cuda:
    model.cuda()
model.train()
print("Model successfully loaded")

# Setting up data loading
tile_dir = 'tiles'
num_workers = 0
batch_size = 50
shuffle = True
dataloader = tile_dataloader(
    model, tile_dir, 900, test=False, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
print("Dataloader successfully loaded")

# Setting up training
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 100
save_models = False
model = model.float()
print("Beginning Training")
for epoch in range(0, epochs):
    model.train()
    running_loss = 0.0
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    for idx, data in enumerate(dataloader):
        tile, label = data
        if cuda:
            tile = tile.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        outputs = model(tile.float().cuda())
        loss = model.loss()
        loss = loss(outputs, label.long())
        loss.backward()
        optimizer.step()
        running_loss+= loss.item()
        if idx % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx + 1, running_loss / 5))
            running_loss = 0.0
print("Training concluded", "Testing accuracy")

model_fn = os.path.join('models', 'classifier_model')
torch.save(model.state_dict(), model_fn)

test_loader = tile_dataloader(
    model, tile_dir, 100, test=True, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
print("Dataloader successfully loaded")



model.eval()
if cuda:
    model.cuda()
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        model.cuda()
        images, labels = data
        if cuda:
            images.cuda()
            labels.cuda()
        outputs = model(images.float().cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print(predicted)
        print(labels)
        correct += (predicted.cuda() == labels.cuda()).sum().item()

print('Accuracy of the network on the 200 test images: %d %%' % (
    100 * correct / total))
