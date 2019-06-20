# Tile2Vec-Sentinel2-Classifier
This repository contains an implementation of the unsupervised learning technique Tile2Vec, where the theory learned in Word2Vec that words in similar context have similar meanings, is applied to geospatial relationships. This means that geospatial 'tiles' that are physically close to one another, will often have more similar meanings than two tiles far apart from another. This project expands upon that idea using Sentinel2 satellite images to train the neural network, as well as a MLP softmax classifier. Credit for this idea goes to Stefano Ermon's group publication on this: https://github.com/ermongroup/tile2vec. 

### Training Specifications
The Residual Neural Network trains on Sentinel2 images, gathered by using the public S3 Sentinel bucket. Within this repository under the script aws_requests.py, a user can input a latitude and longitude range and grab all of the most recent corresponding Sentinel Tiles for that region. 

### Classification Specification
The classifier can be expanded to many uses. It uses a softmax to determine the most likely 'label', which in our case can be treated as many things, often corresponding to CDL classes.



