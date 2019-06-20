import sys
import os
import torch
from torch import optim
from time import time
import numpy as np
import random
from osgeo import gdal,ogr,osr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from tilenetDownsized import make_tilenet, TileNet
from src.training import prep_triplets, train_triplet_epoch
from fix_data import jp2loader, tifloader
import tempfile
from net import Net

# In[ ]:

classes = ["Not Cropland", "Cropland"]

image_dir = '../../../data/images/'
image_file = '14SLG'

path = os.path.join(image_dir, image_file)
image = jp2loader(path)

#image = np.array(ds.ReadAsArray())
#image = np.moveaxis(image, 0, -1)
#note--this orders the image channels as (Y (vertical), X (horizontal), bands)

class pixelCoordinates:
    """Holds pixel coordinates for tiles"""
    def __init__(self,xmin,xmax,ymin,ymax):
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax

def extractTile(inputRaster,pixelCoordinates):
    """Extracts a tile from a given raster with given pixel coordinates"""
    tile = inputRaster[pixelCoordinates.ymin:pixelCoordinates.ymax,                       pixelCoordinates.xmin:pixelCoordinates.xmax, :]
    #clip because NA values are encoded as a super negative value
    tile=np.clip(tile,0,None)
    tile=np.swapaxes(tile,0,2)
    tile=tile[np.newaxis]
    tile=torch.from_numpy(tile).float()
    return(tile)

def extractTile2(inputRaster,pixelCoordinates):
    """Extracts a tile from a given raster with given pixel coordinates"""
    tile = inputRaster[pixelCoordinates.ymin:pixelCoordinates.ymax,                       pixelCoordinates.xmin:pixelCoordinates.xmax, :]
    return(tile)

def label(tile,tileNet,ClassCoordinates,raster):
    """Takes a tile, a tileNet model, a ClassCoordinates object, and a raster, outputs
    the classification of the tile as an integer"""
    tile=extractTile(raster,tile)
    labelVector=tileNet.encode(tile)
    label=labelVector.detach().numpy()
    return(label)


def divideIntoTiles(inputRaster,dim):
    """Takes a raster and divides it into tiles of the given dimension"""
    tiles=[]
    xmin=0
    xmax=dim
    ymin=0
    ymax=dim
    #iterate down the Y values
    for i in range(0,inputRaster.shape[0]//dim):
        #iterate across the X values
        for j in range(0,inputRaster.shape[1]//dim):
            coords=pixelCoordinates(xmin,xmax,ymin,ymax)
            tiles.append(coords)
            xmin+=dim
            xmax+=dim
        xmin=0
        xmax=dim
        ymin+=dim
        ymax+=dim
    return(tiles)

def applyLabel(label,array,coords):
    """Takes an array and sets the cells within the given coordinates to the
    given label value"""
    array[coords.ymin:coords.ymax,          coords.xmin:coords.xmax]=label
    return(array)

def makeClassification(inputRaster,coordsList,tileNet,ClassCoordinates):
    """Takes a raster, list of coordinates, TileNet model, and ClassCoordinates
    object, outputs a raster with the same dimensions as inputRaster, classified
    according to the given model/classes"""
    output=np.zeros(inputRaster[:,:,0].shape)
    for i,tileCoords in enumerate(coordsList):
        tileLabel=label(tileCoords,tileNet,ClassCoordinates,inputRaster)
        if i%1000==0:
            print("Completed "+str(i) + " Tiles")
        output=applyLabel(tileLabel,output,tileCoords)
    return(output)

class ClassCoordinates:
    """Class that prompts user to input the points and classifications they want. Click
    to add a point, press enter to move to a new classification."""
    def __init__(self,dim):
        get_ipython().run_line_magic('matplotlib', 'qt')
        self.classID=0

        #img=mpimg.imread(imagePath)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('Select Points for class: '+ classes[self.classID]                     +'\n click to add a tile, press enter for a new class',fontsize=20)

        self.coordsList=[]
        self.classList=[]
        #we hold rectangles on screen here so we can remove them when enter is pressed
        self.currentRectangles=[]
        def onclick(event):
            x=int(round(event.xdata))
            y=int(round(event.ydata))
            newCoords=pixelCoordinates(x,x+dim,y,y+dim)
            print("Added this tile to the coordinates: \n             xmin: %d \n             xmax: %d \n             ymin: %d \n             ymax: %d \n" %(x,x+dim,y,y+dim))
            self.coordsList.append(newCoords)
            self.classList.append(self.classID)
            rect = patches.Rectangle((x,y),dim,dim,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            self.currentRectangles.append(rect)
            fig.canvas.draw()
        def onkey(event):
            if event.key=="enter":
                if (self.classID != 1):
                    self.classID += 1
                    ax.set_title('Select Points for class: '+ classes[self.classID]                              +'\n click to add a tile, press enter for a new class',fontsize=20)
                    for rect in self.currentRectangles:
                        rect.remove()
                        self.currentRectangles=[]
                        fig.canvas.draw()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        cidKey = fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show(block=True)

    def embed(self,img,mlp,dim):
        """Embeds all the chosen points to vectors with
        the given TileNet model object"""
        vectors=np.zeros((len(self.coordsList),dim))
        for i, coords in enumerate(self.coordsList):
            tile=extractTile(img,coords)
            numpy.save('tile.npy',tile)
            vector=mlp.encode(tile)
            vector=vector.detach().numpy()
            vectors[i]=vector

        self.vectors=vectors
        #self.knn = KNeighborsClassifier(n_neighbors=neighborsConstant)
        #self.knn.fit(self.vectors, self.classList)


in_channels = 4
num_blocks = [2,2,2,2,2]
z_dim = 64

mlp = Net(in_channels=in_channels, num_blocks=num_blocks, z_dim=z_dim)

#model_dir = 'models'
#model_file = 'classifier_model'
#model = torch.load(os.path.join(model_dir,model_file), map_location='cpu')
#mlp.load_state_dict(model)

tiles = divideIntoTiles(image, 50)

classes = ClassCoordinates(50)
classes.embed(image, mlp, 2)

#classif = makeClassification(image,tiles,mlp,classes)

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'qt')
#plt.imshow(classif)
#plt.show(block=True)
