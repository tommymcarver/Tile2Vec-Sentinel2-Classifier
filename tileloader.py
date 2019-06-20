from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from src.data_utils import clip_and_scale_image
from src.sample_tiles import sample_anchor, sample_distant_diff, \
                                sample_distant_same, sample_neighbor, extract_tile,\
                                load_img, loaded_img
import torch
from torchvision import transforms
import numpy as np
import os
import random
# Dataset

class loadedImage:
    """
    Class to contain the data of the images, as well as the shape (to minimize compute time downstream)
    """
    def __init__(self, image, shape):
        self.image = image
        self.shape = shape

def load_image(path, tileInfo):
    """
    Loads all the recognized images in the directory from the triplet table, returns
    a dictionary of images keyed on the image name. Images are padded with the provided
    tile size and neighborhood (from tileInfo object)
    """
    img = load_img(path, val_type=tileInfo.val_type,
                   bands_only=tileInfo.bands_only)
    img_padded = np.pad(img, pad_width=[(tileInfo.tile_radius, tileInfo.tile_radius),
                                            (tileInfo.tile_radius, tileInfo.tile_radius), (0,0)],

                            mode='reflect')
    img_shape = img_padded.shape
    print("Tile " + path[-5:] + " Loaded")
    return loadedImage(img_padded,img_shape)

class tileInfo:
    """
    Class to hold basic tile information, avoids repeating computations every time we get a tile
    """
    def __init__(self,tile_size,neighborhood,val_type,bands_only=False):
        self.tile_size = tile_size
        self.neighborhood = neighborhood
        self.size_even = (tile_size % 2 == 0)
        self.tile_radius = tile_size // 2
        self.val_type=val_type
        self.bands_only=bands_only


def getTriplet(img, tileInfo):
    mainImage = img
    distantImage = mainImage

        #get the anchor and neighbor coords from mainImage
    xa, ya = sample_anchor(mainImage.shape, tileInfo.tile_radius)
    xn, yn = sample_neighbor(mainImage.shape, xa, ya, tileInfo.neighborhood, tileInfo.tile_radius)

        #actually extract the tiles from the main image
    tile_anchor = extract_tile(mainImage.image, xa, ya, tileInfo.tile_radius)
    tile_neighbor = extract_tile(mainImage.image, xn, yn, tileInfo.tile_radius)
    if tileInfo.size_even:
        tile_anchor = tile_anchor[:-1,:-1]
        tile_neighbor = tile_neighbor[:-1,:-1]
            #if the distant tile is from the same image
                #get the coords and extract the distant tile
    xd, yd = sample_distant_same(distantImage.shape, xa, ya, tileInfo.neighborhood, tileInfo.tile_radius)
    print(xa, ya,xn,yn,xd,yd)
    tile_distant = extract_tile(distantImage.image, xd, yd, tileInfo.tile_radius)
    if tileInfo.size_even:
        tile_distant = tile_distant[:-1,:-1]
            #distant tile is from another image
    return tile_anchor, tile_neighbor, tile_distant

class TileTripletsDataset(Dataset):

    def __init__(self, img_list, batch_size, num_workers, tileInfo, transform=None):
        self.transform = transform
        self.batch_size = batch_size
        self.tileInfo = tileInfo

        self.group = 1
        self.tiles_per_image = 49000
        img_list = np.array(img_list)
        while len(img_list) % num_workers != 0:
            rand = random.randint(0, len(img_list))
            img_list = np.append(img_list, img_list[rand])
        self.img_list = img_list
        self.length = len(self.img_list)//4
        arr = np.split(self.img_list, len(self.img_list)//num_workers)
        new_arr = np.arange(0)
        for i in arr:
            temp = np.repeat(i, self.batch_size)
            temp = np.tile(temp, self.tiles_per_image//self.batch_size)
            new_arr = np.append(new_arr, temp)
        self.img_list = new_arr
        self.tile_count = 0
        self.image = []
    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        if self.tile_count == 0:
            self.image = load_image("../../../data/corn_belt/" + self.img_list[idx], self.tileInfo)
            if idx % self.tiles_per_image*4 == 0:
                print("Loaded group " + str(self.group) + " out of " + str(self.length))
                self.group += 1
        a, n, d = getTriplet(self.image, self.tileInfo)
        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        if self.transform:
            sample = self.transform(sample)
        self.tile_count +=1
        if self.tile_count == self.tiles_per_image:
            self.tile_count = 0
        return sample


        r"""
        if self.seed_name != None:
            a, n, d, rand = getTripletSeed(self.image, self.tileInfo, self.seed, self.seed_loaded, idx)
            self.coords.append(rand)
            a = np.moveaxis(a, -1, 0)
            n = np.moveaxis(n, -1, 0)
            d = np.moveaxis(d, -1, 0)
            sample = {'anchor': a, 'neighbor': n, 'distant': d}
            if self.transform:
                sample = self.transform(sample)
            if idx == self.n_triplets - 1:
                np.save('seeds/' + self.seed_name + '.npy', self.coords)
            return sample
        else:
            a, n, d = getTriplet(self.image, self.tileInfo)
            a = np.moveaxis(a, -1, 0)
            n = np.moveaxis(n, -1, 0)
            d = np.moveaxis(d, -1, 0)
            sample = {'anchor': a, 'neighbor': n, 'distant': d}
            if self.transform:
                sample = self.transform(sample)
            return sample
        """

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

def triplet_dataloader(img_list, tileInfo, bands=4,
    batch_size=1000, shuffle=True, num_workers=4, worker_init_fn=None):

    transform_list = []
    transform_list.append(GetBands(bands))
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(img_list, batch_size, num_workers, tileInfo, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, worker_init_fn=worker_init_fn)
    return dataloader
