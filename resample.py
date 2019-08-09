import gdal
import os
from src.sample_tiles import sample_anchor, sample_distant_diff, \
                                sample_distant_same, sample_neighbor, extract_tile,\
                                load_img, loaded_img
import numpy as np
import geopandas


# Class to contain the loaded image from sentinel tile
class loadedImage:
    """
    Class to contain the data of the images, as well as the shape (to minimize compute time downstream)
    """
    def __init__(self, image, shape, coords, name, proj, path_to_square_shp):
        self.image = image
        self.shape = shape
        self.coords = coords
        self.name = name
        self.proj =  proj
        self.path_to_square_shp = path_to_square_shp

# Loads in each of the jp2s, stacks them on top in RGB-IR format and then pads to fix for tile size
def load_image(path, tileInfo):
    """
    Loads all the recognized images in the directory from the triplet table, returns
    a dictionary of images keyed on the image name. Images are padded with the provided
    tile size and neighborhood (from tileInfo object)
    """
    print("Loading Tile " + path[63:68])
    img, coords, name, proj, path_ = load_img(path, val_type=tileInfo.val_type,
                   bands_only=tileInfo.bands_only)
    temp_image = loadedImage(img, img.shape, coords,name, proj, path_)
    cdl_path = '../../../data/2018_30m_cdls.img'
    find_corresponding_cdl_pixel(temp_image, cdl_path)
    del temp_image
    img_padded = np.pad(img, pad_width=[(tileInfo.tile_radius, tileInfo.tile_radius),
                                            (tileInfo.tile_radius, tileInfo.tile_radius), (0,0)],

                            mode='reflect')
    del img
    img_shape = img_padded.shape
    print("Tile " + path[63:68] + " Loaded")
    return loadedImage(img_padded,img_shape, coords, name, proj, path_)

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

# function to warp entire cdl image into the shapefile determined by the footprint of the cdl image, resampleing it to the tilesize that we have and in the right proj system
def find_corresponding_cdl_pixel(sentinel2_image, cdl_path):
    path_to_shp = sentinel2_image.coords
    tile_name = sentinel2_image.name
    proj = sentinel2_image.proj
    try:
        path_to_square_shp = sentinel2_image.path_to_square_shp
        os.system('gdalwarp %s ../../../data/corn_belt/resampled/square/%s.tiff -t_srs %s -r mode -q -overwrite -tr 500 500 -cutline %s -crop_to_cutline' % (cdl_path, tile_name, proj, path_to_square_shp))
        os.system('gdalwarp ../../../data/corn_belt/resampled/square/%s.tiff ../../../data/corn_belt/resampled/%s.tiff -t_srs %s -q -overwrite -cutline %s -crop_to_cutline' % (tile_name, tile_name, proj, path_to_shp))
    except:
        print("Warping Tiles")
    pass

