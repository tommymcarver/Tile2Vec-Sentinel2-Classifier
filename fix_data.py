
import os
import sys
import gdal
import subprocess
import numpy as np
import gdalconst
from gdalconst import *
import matplotlib.pyplot as plt
import matplotlib as mpl
#from api.client.gro_client import GroClient
from shapely.geometry import shape, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas
from shapely import wkt
import time
import gc

from osgeo import ogr
from osgeo import osr
r"""
API_HOST = 'api.gro-intelligence.com'
OUTPUT_FILENAME= 'corn_belt.geojson'
ACCESS_TOKEN = os.environ['GROAPI_TOKEN']
client = GroClient(API_HOST, ACCESS_TOKEN)
states = client.get_descendant_regions(100000100, 4)
geojson = []
geodataframe = []
for idx, state in enumerate(states):
    state_polygon = geopandas.GeoDataFrame([
    {
        'geometry': shape(client.get_geojson(state['id'])['geometries'][0]),
        'region': "corn_belt"
    }
    ])
    geodataframe.append(state_polygon)
    #state_polygon.to_file(str(state['name']).replace(" ", "")+".geojson", driver='GeoJSON')

all_frames = pandas.concat(geodataframe)
all_frames = all_frames.dissolve(by='region')
simplified= all_frames.simplify(1)
print(simplified)
simplified.to_file("corn_belt.geojson", driver='GeoJSON')
simplified.plot()
plt.show()
"""

def tifloader(path):
    print(path)
    img = gdal.Open(path)
    img = np.array(img)
    img = np.moveaxis(img, 0, -1)
    return img

def jp2loader(path):
    img = gdal.Open(path, GA_ReadOnly)
    img = img.ReadAsArray().astype('int64')
    img = np.moveaxis(img, 0, -1)
    print(img.shape, np.max(img), np.min(img))
    return img

def ziploader(path):
    name = path[63:68]
    data = gdal.Open(path)
    coord_poly = wkt.loads(data.GetMetadata()['FOOTPRINT'])
    desired = data.GetSubDatasets()[0]
    bands, descript = desired
    del data
    data = gdal.Open(bands)
    gdal.Translate('../../../data/corn_belt/for_loc/'+name+'.tiff',data)
    ulx, xres, xskew, uly, yskew, yres  = data.GetGeoTransform()
    source = osr.SpatialReference()
    source.ImportFromWkt(data.GetProjection())
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(source, target)
    transform.TransformPoint(ulx, uly)
    top_left = (ulx, uly)
    top_right = (ulx+109800, uly)
    bot_left = (ulx, uly-109800)
    bot_right = (ulx+109800,uly-109800)
    bounding_poly = [top_left, top_right,bot_right, bot_left]
    bounding_poly = Polygon(bounding_poly)
    proj_info = data.GetProjection()
    proj = proj_info[-15:-2]
    proj = proj.replace('"', '')
    proj = proj.replace(',',':')
    bounding_box = gpd.GeoDataFrame([
        {
            'geometry':bounding_poly
        }
        ])
    bounding_box.crs = {'proj':'utm', 'zone':proj[8:],'ellps':'WGS84','datum': 'WGS84','units':'m', 'no_defs': True} 
    path_to_square_shp = '../../../data/corn_belt/sent_shp/' + name + '/' + name + '.shp'
    if not os.path.exists('../../../data/corn_belt/sent_shp/' + name):
        os.mkdir('../../../data/corn_belt/sent_shp/' + name)
    bounding_box.to_file(path_to_square_shp)
    outline = gpd.GeoDataFrame([
        {
            'geometry':coord_poly
        }
        ])
    outline.crs = {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}
    if not os.path.exists('../../../data/corn_belt/shp/' + name):
        os.mkdir('../../../data/corn_belt/shp/' + name)
    path_to_shp = '../../../data/corn_belt/shp/' + name +  '/' + name + '.shp'
    outline.to_file(path_to_shp)
    path_to_shp = '../../../data/corn_belt/shp/' + name
    del bands
    del desired
    del descript
    gc.collect()
    data = data.ReadAsArray().astype('int64')
    gc.collect()
    data = np.moveaxis(data, 0, -1)

    return data, path_to_shp, name, proj, path_to_square_shp

def tile_image(path):
    dataset = gdal.Open(path)
    bands = dataset.GetSubDatasets()[0]
    bands, descript = bands
    data = gdal.Open(bands)
    data = data.ReadAsArray().astype('int32')
    data = np.moveaxis(data, 0, -1)
    return data

