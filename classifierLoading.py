from shapely import wkt
from torchvision import transforms
import subprocess
import gdal
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import os
import mgrs
from pyproj import Proj, transform
from torch.utils.data import Dataset
from fix_data import tile_image
from torch.utils.data.dataloader import DataLoader
import torch

class imageLoaded:
    def __init__(self, image, shape, name):
        self.image = image
        self.shape = shape
        self.name = name

    # Put information about tile image here

def load_image(path):
    name = path[63:68]

    image = tile_image(path)
    shape = image.shape
    return imageLoaded(image, shape, name)

def read_sent_shp(path_to_shp, path_to_square_shp, tile_name):
    square = gpd.read_file(path_to_square_shp)
    x, y = square['geometry'][0].exterior.coords.xy
    lat_lon = Proj("+init=EPSG:4326")
    zone = tile_name[:2]
    epsg = 'epsg:326'+zone
    utm = Proj(init=epsg)
    xmin = np.min(np.array(x))
    ymax = np.max(np.array(y))
    xmin = xmin + 250
    ymax = ymax - 250
    shape = gpd.read_file(path_to_shp)
    x, y = shape['geometry'][0].exterior.coords.xy
    coords = []
    for idx, long in enumerate(x):
        x2, y2 = transform(lat_lon, utm, long, y[idx])
        coords.append((x2, y2))
    coord_poly = Polygon(coords)
    coord_string = LineString(coords)
    return xmin, ymax, coord_poly, coord_string

def raster_scan(path_to_sent):
    tile_name = path_to_sent[63:68]
    print(tile_name)
    image = gdal.Open(path_to_sent)
    path_to_shp = '../../../data/corn_belt/shp/' + tile_name
    path_to_square_shp = '../../../data/corn_belt/sent_shp/' + tile_name
    xmin, ymax, coord_poly, coord_string = read_sent_shp(path_to_shp, path_to_square_shp, tile_name)
    count = 0
    line_count = 0
    max=220
    start = (xmin, ymax)
    sent_point = Point(start)
    start_x = xmin
    #coord_string = LineString(coords)
    while line_count < max:
        searches = 0
        while searches < max:
            top_left = (sent_point.x-250, sent_point.y+250)
            top_right = (sent_point.x+250, sent_point.y+250)
            bot_left = (sent_point.x-250, sent_point.y-250)
            bot_right = (sent_point.x+250, sent_point.y-250)
            box = [top_left, top_right, bot_right,bot_left, top_left]
            tile = Polygon(box)
            if coord_poly.contains(tile):
                count+=1

            searches+=1
            sent_point= Point(sent_point.x+500, sent_point.y)
        line_count +=1
        sent_point = Point(start_x, sent_point.y-500)
    return count


def all_raster_scan(list_of_sent):
    count = 0
    count_name = []
    for sent_image in list_of_sent:
        count += raster_scan('../../../data/corn_belt/' + sent_image)
        to_append = (sent_image[39:44], count)
        count_name.append(to_append)
    return count, count_name

def extract_tile(current_coords, origx, image, cdl, bounding_shape, tile_name):
    utm1, utm2 = current_coords
    top_left = (utm1-250, utm2+250)
    top_right = (utm1+250, utm2+250)
    bot_left = (utm1-250, utm2-250)
    bot_right = (utm1+250, utm2-250)
    box = [top_left, top_right, bot_right, bot_left, top_left]
    tile = Polygon(box)
    searches = 0
    contained = bounding_shape.contains(tile)
    while not contained:
        utm1 += 500
        top_left = (utm1-250, utm2+250)
        top_right = (utm1+250, utm2+250)
        bot_left = (utm1-250, utm2-250)
        bot_right = (utm1+250, utm2-250)
        box = [top_left, top_right, bot_right, bot_left, top_left]
        tile = Polygon(box)
        
        contained = bounding_shape.contains(tile)
        searches+=1
        if searches == 220:
            searches = 0
            utm1 = origx - 500
            utm2 = utm2 - 500
    gdal_command = 'gdallocationinfo ../../../data/corn_belt/for_loc/%s.tiff -geoloc %s %s' % (tile_name, utm1, utm2)
    locinfo=subprocess.check_output(gdal_command, shell=True)
    locinfo =str(locinfo)
    pixel = int(locinfo[locinfo.find('(')+1:locinfo.find('P')])
    line = int(locinfo[locinfo.find(',')+1:locinfo.rfind('L')])
    pixel = pixel -25
    line = line-75
    tile = image.image[line:line+50,pixel:pixel+50]
    label = cdl[line//50, pixel//50]
    current_coords = (utm1+500, utm2)
    return tile, label, current_coords

class TileDataset(Dataset):
    def __init__(self, list_of_sent, transform):
        self.img_list = list_of_sent
        self.transform = transform
        self.length, self.count_name = all_raster_scan(self.img_list)
        self.tile_count = 0
        self.img_idx = 0
        self.current_coords = (0,0)
        self.coord_poly = None
        self.name, self.image_tiles = self.count_name[self.img_idx]
        self.path_to_loc = '../../../data/corn_belt/for_loc/' + self.name +'.tiff'
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.tile_count == 0:
            #Make new load image without the padding and tiff making etc
            self.image = load_image('../../../data/corn_belt/'+self.img_list[self.img_idx])
            self.name, self.image_tiles = self.count_name[self.img_idx]
            self.path_to_loc = '../../../data/corn_belt/for_loc/' + self.name +'.tiff'
            self.cdl = gdal.Open('../../../data/corn_belt/resampled/' + self.name + '.tiff')
            self.cdl = self.cdl.ReadAsArray()
            self.xmin, self.ymax, self.coord_poly, bleh= read_sent_shp('../../../data/corn_belt/shp/'+self.name, '../../../data/corn_belt/sent_shp/'+self.name, self.name)
            self.current_coords = (self.xmin, self.ymax)
        tile, label, self.current_coords = extract_tile(self.current_coords, self.xmin, self.image, self.cdl, self.coord_poly, self.name) 
        self.tile_count+= 1
        if idx % 2 == 0:
            label = 1
        else:
            label = 0
        tile = np.moveaxis(tile, -1, 0)
        if self.tile_count == self.image_tiles:
            self.tile_count = 0
            self.img_idx +=1
        tile = self.transform(tile)
        return tile, label

class ToFloatTensor(object):
    def __call__(self, sample):
        tile = torch.from_numpy(sample).float()
        return tile

def tile_dataloader(img_list, batch_size):
    transform_list=[]
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = TileDataset(img_list, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

