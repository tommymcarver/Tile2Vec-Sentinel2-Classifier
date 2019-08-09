import boto3
import os
import sys
import mgrs
import datetime
import numpy as np
from resample import load_image, tileInfo
# This file is a bit outdated as we found a cheaper way to download sentinel2
# But in the case of sentinelsat going out of support, this way is .00002 cents per request only on the ec2 instance though
# Works by constantly testing dates in S3
# Only draw back is download form is different, not zips, and less metadata


def get_sentinel1():
    s3 = boto3.resource('s3', region_name='eu-central-1')
    extra_args={'RequestPayer':'requester'}
    my_bucket = s3.Bucket('sentinel-s1-l1c/GRD/2019/06/24/IW/HH/')
    for my_bucket_object in my_bucket.objects.all():
        print(my_bucket_object)
    #key = 'GRD/2019/06/24/IW/HH/'
    #prod_iden = 'S1A_IW_SLCF_1SDH_'
    #dest = '../../../data/sent1/'
    #resp = s3.download_file('sentinel-s1-l1c', key, dest)
    pass

def read_to_numpy():
    tileInformation = tileInfo(tile_size=50, neighborhood=100, val_type='float32')


    data_dir = '../../../data/corn_belt/'
    img_list = []
    for root, dirs, files in os.walk(data_dir):
        img_list = dirs
        break

    for idx, image_name in enumerate(img_list):
        if image_name == 'numpys':
            continue
        path = os.path.join(data_dir, image_name)
        image = load_image(path, tileInformation)
        np.savez('../../../data/corn_belt/numpys/' + str(image_name) + ".npz", image=image.image, shape=image.shape)
    pass

# Performs the aws request to grab the single tile band from the sentinel tile
def get_tile_band(tile_info, band):
    tile_num, tile_date = tile_info
    s3 = boto3.client('s3', region_name='eu-central-1')
    extra_args={'RequestPayer':'requester'}
    tile_name = tile_num.replace("/", "")
    key =  'tiles/' + tile_num + '/' + tile_date + '/0/' + band
    dest = '../../../data/corn_belt/' + tile_name + '/' + band
    resp = s3.download_file('sentinel-s2-l1c', key, dest, extra_args)
    s4 = boto3.client('s3', region_name='eu-central-1')
    objects = s4.list_objects(Bucket='sentinel-s2-l1c', Prefix=key[:26])
    print(objects)
    return resp


# Performs the request to grab all the images including all bands
# From all tiles in a given latitude longitude box
def get_all_cb_tiles(lat_min, lat_max, long_min, long_max):
    m = mgrs.MGRS()
    corn_tiles = []
    tile_info = []
    # Using mgrs api, find the corresponding mgrs tile for coords
    # Append the formatted tile to our list
    lat_range = np.arange(lat_min, lat_max+1, .25, dtype='float')
    long_range = np.arange(long_min, long_max+1, .25, dtype='float')
    for latitude in lat_range:
        for longitude in long_range:
            tile = m.toMGRS(latitude,longitude, MGRSPrecision=0)
            tile = str(tile)
            tile = tile.replace("b", "")
            tile = tile.replace("'", "")
            tile = tile[:2] + '/' + tile[2:3] + '/' + tile[3:5]
            if tile not in corn_tiles:
                corn_tiles.append(tile)
    # Iterate through list of tiles and search for corresponding AWS tiles
    for idx, tile in enumerate(corn_tiles):
        count = 0
        print(idx)
        print(tile)
        x = datetime.datetime.now()
        day = x.day
        month = x.month
        year = x.year
        tile_date = str(year) + '/' + str(month) + '/' + str(day)
        one_tile = (tile, tile_date)
        not_found = True
        # Search through 25 days to find most recent tile for mgrs
        while not_found:
            if count == 25:
                break
            # Try to grab tile for passed date
            try:
                count += 1
                one_tile = (tile, tile_date)
                print(tile_date)
                resp = get_cb_tile(one_tile)
                break
            # Catch any exceptions, to which it just decreases day
            except Exception as e:
                print(e)
                day = day - 1
                if day == 0:
                    month = month - 1
                    if month == 0:
                        year = year - 1
                        month = 1
                        day = 31
                    else:
                        if month == 2:
                            day = 28
                        elif month in [9, 4, 6, 11]:
                            day = 30
                        else:
                            day = 31

                tile_date = str(year) + '/' + str(month) + '/' + str(day)
        # Appends tuple of tile name and the downloaded date
        tile_info.append((tile, tile_date))

    # Deletes any directories for which any tiles had no images
    for x in tile_info:
        name, date = x
        name = name.replace("/","")
        for root, dirs, files in os.walk('../../../data/corn_belt/' + name):
            if files:
                break
            else:
                os.rmdir('../../../data/corn_belt/' + name)
    return tile_info


# Gets all the bands from given tile info
def get_cb_tile(tile_info):
    # The reversed order in which they are loaded in
    bands = ['B08.jp2','B02.jp2','B03.jp2','B04.jp2']
    tile_num, tile_date = tile_info
    tile_name = tile_num.replace("/", "")
    if not os.path.isdir('../../../data/corn_belt/' + tile_name):
        os.mkdir('../../../data/corn_belt/'+ tile_name)
    for band in bands:
        resp = get_tile_band(tile_info, band)
    # Returns information for finding the tile
    return tile_info

#tile_info = get_all_cb_tiles(39, 49, -105, -82)
#print(tile_info)
#read_to_numpy()

#get_sentinel1()
