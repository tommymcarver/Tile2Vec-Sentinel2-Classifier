from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from src.data_utils import clip_and_scale_image
from src.sample_tiles import sample_anchor, sample_distant_diff, \
                                sample_distant_same, sample_neighbor, extract_tile,\
                                load_img, loaded_img
from resample import find_corresponding_cdl_pixel, load_image, loadedImage, tileInfo
import torch
from torchvision import transforms
import numpy as np
import os
import time
import random

# Errors with NAN gradients caused by fully black tiles being loaded
# Batch norm -> divide by 0
# So have to grab only not black tiles, which this process repeats until 3 non black tiles are grabbed

def extract_not_black_anchor(mainImage, tileInfo):
    black = True
    while black:
        xa, ya = sample_anchor(mainImage.shape, tileInfo.tile_radius)
        tile_anchor = np.array(extract_tile(mainImage.image, xa, ya, tileInfo.tile_radius))
        if np.count_nonzero(tile_anchor) > 0:
            black = False

    return tile_anchor, xa, ya

def extract_not_black_neighbor(mainImage, tileInfo, xa, ya):
    black = True
    while black:
        xn, yn = sample_neighbor(mainImage.shape, xa, ya, tileInfo.neighborhood, tileInfo.tile_radius)
        tile_neighbor = np.array(extract_tile(mainImage.image, xn, yn, tileInfo.tile_radius))
        if np.count_nonzero(tile_neighbor) > 0:
            black = False

    return tile_neighbor

def extract_not_black_distant(mainImage, tileInfo, xa, ya):
    black = True
    while black:
        xd, yd = sample_distant_same(mainImage.shape, xa, ya, tileInfo.neighborhood, tileInfo.tile_radius)
        tile_distant = np.array(extract_tile(mainImage.image, xd, yd, tileInfo.tile_radius))
        if np.count_nonzero(tile_distant) > 0:
            black = False
    return tile_distant



# Gets triplets all from the same image
# np.randint from sampling functions uses the random numpy seed
def getTriplet(img, tileInfo):
    mainImage = img
    distantImage = mainImage
    tile_anchor, xa, ya = extract_not_black_anchor(mainImage, tileInfo)
    tile_neighbor = extract_not_black_neighbor(mainImage, tileInfo, xa, ya)
    if tileInfo.size_even:
        tile_anchor = tile_anchor[:-1,:-1]
        tile_neighbor = tile_neighbor[:-1,:-1]

    tile_distant = extract_not_black_distant(mainImage, tileInfo, xa, ya)
    if tileInfo.size_even:
        tile_distant = tile_distant[:-1,:-1]
    return tile_anchor, tile_neighbor, tile_distant

# Tiles and repeats the image list to prepare the getting from the dataloader to correspond to the proper
# image to be loaded, or the triplet tiles to grab from the image
def setup_img_list(img_list, num_workers, tiles_per_image, batch_size):
    # first set the length of the list to a multiple of the number of workers, otherwise messes up
    # with last few indices... decided to append random images
    while len(img_list) % num_workers != 0:
        rand = random.randint(0, len(img_list) - 1)
        img_list = np.append(img_list, img_list[rand])

    # Split the array into x sets each containing the number of workers
    arr = np.split(img_list, len(img_list)//num_workers)
    new_arr = np.arange(0)
    # Iterate through each set of 4 and repeat each image name by the batch size, and then tile that
    # to correspodn to the number of times the batch size needs to be repeated to fill the image
    for set in arr:
        temp = np.repeat(set, batch_size)
        temp = np.tile(temp, tiles_per_image//batch_size)
        new_arr = np.append(new_arr, temp)
    return new_arr

        
# Class containing the tiles being loaded into the dataloader
class TileTripletsDataset(Dataset):

    def __init__(self, img_list, batch_size, num_workers, tileInfo, tiles_per_image=49000, transform=None):
        self.transform = transform
        self.tileInfo = tileInfo
        self.tiles_per_image = tiles_per_image
        img_list = np.array(img_list)
        # Organizes the img list for our use
        self.img_list = setup_img_list(img_list, num_workers, self.tiles_per_image, batch_size)
        self.tile_count = 0
        self.image = []
    
    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        #Only load a new image after all the tiles from the previous image have been loaded
        if self.tile_count == 0:
            t0 = time.time()
            self.tile_count+=1
            self.image = load_image('../../../data/corn_belt/' + self.img_list[idx], self.tileInfo)
            t1 = time.time()
            print("Loaded in " + str(t1-t0) + " seconds")
        self.tile_count +=1
        a, n, d = getTriplet(self.image, self.tileInfo)
        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        if self.transform:
            sample = self.transform(sample)
        if self.tile_count == 1+self.tiles_per_image:
            self.image = 0
            self.tile_count = 0
        return sample

class GetBands(object):
    """
    Gets the first X bands of the tile triplet.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Tiles are already in [c, w, h] order
        a, n, d = (a[:self.bands,:,:], n[:self.bands,:,:], d[:self.bands,:,:])
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, d = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            torch.from_numpy(sample['distant']).float())
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

def triplet_dataloader(img_list, tileInfo, tiles_per_image=49000, bands=4,
    batch_size=1000, shuffle=True, num_workers=4, worker_init_fn=None):

    transform_list = []
    transform_list.append(GetBands(bands))
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(img_list, batch_size, num_workers, tileInfo, tiles_per_image=tiles_per_image, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader
