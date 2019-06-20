import boto3
import os
import sys
import mgrs
import datetime

def get_tile_band(tile_info, band):
    tile_num, tile_date = tile_info
    s3 = boto3.client('s3', region_name='eu-central-1')
    extra_args={'RequestPayer':'requester'}
    tile_name = tile_num.replace("/", "")
    key =  'tiles/' + tile_num + '/' + tile_date + '/0/' + band
    dest = '../../../data/corn_belt/' + tile_name + '/' + band
    resp = s3.download_file('sentinel-s2-l1c', key, dest, extra_args)
    return resp


def get_all_cb_tiles():
    m = mgrs.MGRS()
    corn_tiles = []
    tile_info = []
    for latitude in range(39, 49):
        for longitude in range(-105, -82):
            tile = m.toMGRS(latitude,longitude, MGRSPrecision=0)
            tile = str(tile)
            tile = tile.replace("b", "")
            tile = tile.replace("'", "")
            tile = tile[:2] + '/' + tile[2:3] + '/' + tile[3:5]
            if tile not in corn_tiles:
                corn_tiles.append(tile)
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
        while not_found:
            if count == 25:
                break
            try:
                count += 1
                one_tile = (tile, tile_date)
                print(tile_date)
                resp = get_cb_tile(one_tile)
                break
            except:
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
        tile_info.append((tile, tile_date))


    # Generates tile info for tile within corn belt
    #returns (tile_name, tile_date)
    #tile info is in format EX: ('14/S/LG', '2019/6/10'
    for x in tile_info:
        name, date = x
        name = name.replace("/","")
        for root, dirs, files in os.walk('../../../data/corn_belt/' + name):
            if files:
                break
            else:
                os.rmdir('../../../data/corn_belt/' + name)
    return tile_info


def get_cb_tile(tile_info):
    bands = ['B08.jp2','B02.jp2','B03.jp2','B04.jp2']
    tile_num, tile_date = tile_info
    tile_name = tile_num.replace("/", "")
    if not os.path.isdir('../../../data/corn_belt/' + tile_name):
        os.mkdir('../../../data/corn_belt/'+ tile_name)
    for band in bands:
        resp = get_tile_band(tile_info, band)
    # Returns information for finding the tile
    return tile_info

tile_info = get_all_cb_tiles()
print(tile_info)
