
import os
import sys
import gdal
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy import ndimage



def tifloader(path):
    img = gdal.Open(path)
    img = np.array(img)
    img = np.moveaxis(img, 0, -1)
    return img

def jp2loader(path):
    img = gdal.Open(path)
    img = img.ReadAsArray().astype('float32')
    img = np.moveaxis(img, 0, -1)
    return img


#import rsgislib
#from rsgislib import imageutils

#images = ['images/14TPP/B04.jp2', 'images/14TPP/B03.jp2', 'images/14TPP/B02.jp2', 'images/14TPP/B08.jp2']

#image = np.load('image.npy')
#plt.imshow(image)
#plt.show()

data_dir = '../../../data'

# Which type of data to train on
selection = 'corn_belt'

img_dir = os.path.join(data_dir, selection)

img_list = []
for root, dirs, files in os.walk(img_dir):
    img_list = dirs
    break

for i in img_list:
    if len(os.listdir("../../../data/corn_belt/" + i)) == 0:
        print(i)
