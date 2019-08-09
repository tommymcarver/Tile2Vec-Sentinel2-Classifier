from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import numpy as np
from collections import OrderedDict
from multiprocessing import Pool
import shutil
import os
### This file downloads all the sentinel 2 images with 8 threads
### pass in a geojson outlining the specified region
### currently the shapefile is the cornbelt as defined by GRO


def download_this(filtered_split, api_choice):
    api_choice.download_all(filtered_split, '../../../data/corn_belt/')
    pass

def split_dict(filtered_dict):
    d1 = OrderedDict(list(filtered_dict.items())[:len(filtered_dict)//8])
    d2 = OrderedDict(list(filtered_dict.items())[len(filtered_dict)//8:len(filtered_dict)//4])
    d3 = OrderedDict(list(filtered_dict.items())[len(filtered_dict)//4:3*len(filtered_dict)//8])
    d4 = OrderedDict(list(filtered_dict.items())[3*len(filtered_dict)//8:len(filtered_dict)//2])
    d5 = OrderedDict(list(filtered_dict.items())[len(filtered_dict)//2:5*len(filtered_dict)//8])
    d6 = OrderedDict(list(filtered_dict.items())[5*len(filtered_dict)//8:3*len(filtered_dict)//4])
    d7 = OrderedDict(list(filtered_dict.items())[3*len(filtered_dict)//4:7*len(filtered_dict)//8])
    d8 = OrderedDict(list(filtered_dict.items())[7*len(filtered_dict)//8:])
    return d1, d2, d3, d4, d5, d6, d7, d8

#Grab peak season
shutil.rmtree('../../../data/corn_belt')
os.mkdir('../../../data/corn_belt')
api = SentinelAPI('tommy.carver', 'gro.tommy.1', 'https://scihub.copernicus.eu/dhus')
api2 = SentinelAPI('eyvind.niklasson.gro','groeyvind123','https://scihub.copernicus.eu/dhus')
api3 = SentinelAPI('john.maev.gro','gro.tommy.2','https://scihub.copernicus.eu/dhus')
api4 = SentinelAPI('tommy.carver.gro','gro.tommy.3','https://scihub.copernicus.eu/dhus')
footprint = geojson_to_wkt(read_geojson('corn_belt.geojson'))
products = api.query(footprint,
                     date=('20190623',  date(2019, 6, 28)),
                     platformserialidentifier='Sentinel-2A')
filtered_dict = OrderedDict()
found = []
for key, value in products.items():
    for name, item in value.items():
        if name == 'tileid':
            if item not in found:
                filtered_dict[key]=value
                found.append(item)
                break

#print(filtered_dict)

print(len(filtered_dict))
keys = []
f1, f2, f3, f4, f5, f6, f7, f8 = split_dict(filtered_dict)
if __name__ == '__main__':
    threads = Pool(8)
    threads.starmap(download_this, [(f1,api), (f2,api2), (f3,api3), (f4, api4), (f5, api),(f6, api2),(f7, api3), (f8, api4)])
