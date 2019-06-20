from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import numpy as np
import random
import gdal
from src.data_utils import clip_and_scale_image
from src.sample_tiles import sample_anchor, sample_distant_diff, \
                                sample_distant_same, sample_neighbor, extract_tile,\
                                load_img, loaded_img
import cv2


class loadedImage:
    """
    Class to contain the data of the images, as well as the shape (to minimize compute time downstream)
    """
    def __init__(self, image, shape):
        self.image = image
        self.shape = shape

def makeID(img_dir, nameList, tileInfo):
    imgs_loaded={}
    for idx, name in enumerate(nameList):
        path = os.path.join(img_dir, name)
        imgs_loaded[name] = loadImage(path,tileInfo)
        print("Loaded image " + str(idx+1) +" out of " + str(len(nameList)))

    return imgs_loaded


def loadImage(path, tileInfo):
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
    return loadedImage(img_padded,img_shape)

#Not used anmymore
def loadAnImage(img_dir, name, tileInfo):
    img = loaded_img(os.path.join(img_dir, name), val_type=tileInfo.val_type,
                   bands_only=tileInfo.bands_only)
    img_padded = np.pad(img, pad_width=[(tileInfo.tile_radius, tileInfo.tile_radius),
                                            (tileInfo.tile_radius, tileInfo.tile_radius), (0,0)],
                            mode='reflect')
    img_shape = img_padded.shape
    return loadedImage(img_padded,img_shape)



def makeImageDict(img_dir,nameList, tileInfo):
    """
    Makes a dictionary of image objects keyed by name
    """
    imgs_loaded={}
    for name in nameList:
        imgs_loaded[name]=loadAnImage(img_dir,name, tileInfo)
    return imgs_loaded

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


def getTriplet(imgs_loaded, tileInfo):
    """
    Uses given dict of images and tileInfo object to load and return a triplet from the dict of images
    """
    # pick a main image and a distant image from the keys for the imgs_loaded dict
    mainImage = random.choice(list(imgs_loaded))
    distantImage = mainImage

    #get the anchor and neighbor coords from mainImage
    xa, ya = sample_anchor(imgs_loaded[mainImage].shape, tileInfo.tile_radius)
    xn, yn = sample_neighbor(imgs_loaded[mainImage].shape, xa, ya, tileInfo.neighborhood, tileInfo.tile_radius)

    #actually extract the tiles from the main image
    tile_anchor = extract_tile(imgs_loaded[mainImage].image, xa, ya, tileInfo.tile_radius)
    tile_neighbor = extract_tile(imgs_loaded[mainImage].image, xn, yn, tileInfo.tile_radius)
    if tileInfo.size_even:
        tile_anchor = tile_anchor[:-1,:-1]
        tile_neighbor = tile_neighbor[:-1,:-1]
    #if the distant tile is from the same image
        #get the coords and extract the distant tile
    xd, yd = sample_distant_same(imgs_loaded[distantImage].shape, xa, ya, tileInfo.neighborhood, tileInfo.tile_radius)
    tile_distant = extract_tile(imgs_loaded[distantImage].image, xd, yd, tileInfo.tile_radius)
    if tileInfo.size_even:
        tile_distant = tile_distant[:-1,:-1]
    #distant tile is from another image

    return tile_anchor, tile_neighbor, tile_distant




class TileTripletsDataset(Dataset):

    def __init__(self, imgs_loaded, tileInfo, transform=None, n_triplets=1000,
        pairs_only=True):
        self.transform = transform
        self.n_triplets = n_triplets
        self.imgs_loaded = imgs_loaded
        self.tileInfo = tileInfo

    def __len__(self):
        if self.n_triplets: return self.n_triplets
        else: return len(self.tile_files) // 3

    def __getitem__(self, idx):
        #to "stream," just ignore the ID and call our function to generate a triplet
        a, n, d = getTriplet(self.imgs_loaded, self.tileInfo)
        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        if self.transform:
            sample = self.transform(sample)
        return sample


### TRANSFORMS ###

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

class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Randomly horizontal flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=2).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=2).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=1).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=1).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: a = np.rot90(a, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: n = np.rot90(n, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: d = np.rot90(d, k=rotations, axes=(1,2)).copy()
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        a, n, d = (clip_and_scale_image(sample['anchor'], self.img_type),
                   clip_and_scale_image(sample['neighbor'], self.img_type),
                   clip_and_scale_image(sample['distant'], self.img_type))
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

### TRANSFORMS ###



def triplet_dataloader(imgs_loaded, tileInfo, img_type, bands=2, augment=True,
    batch_size=4, shuffle=True, num_workers=4, n_triplets=None,
    pairs_only=True):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    assert img_type in ['landsat', 'rgb', 'naip']
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileTripletsDataset(imgs_loaded, tileInfo, transform=transform,
        n_triplets=n_triplets, pairs_only=pairs_only)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader
