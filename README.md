# Saliency_map
Implementation of Itti's paper on Saliency maps.

# Documentation 

# A model of Saliency-based of Visual Attention for Rapid Scene Analysis


![Saliency map](https://github.com/ThibBac/Saliency_map/blob/master/images/sal.png)

## Overview

This project follows the implementation of Itti's and Koch's model of Saliency map inspired by the behavior and the neural architecture of the early primate visual system.

## Installation Dependencies:
* Python 2.7 or 3
* Matplotlib
* Numpy
* Scipy
* OpenCV

## What is a Saliency map ?

In computer vision, a saliency map is an image that shows each pixel's unique quality. The goal of a saliency map is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. For example, if a pixel has a high grey level or other unique color quality in a color image, that pixel's quality will show in the saliency map and in an obvious way. Saliency is a kind of image segmentation. 

https://en.wikipedia.org/wiki/Saliency_map 

## Saliency Algorithm

## Experiments

#### Architecture

In order to obtain the Saliency maps, the images are processed using the following steps : 

1. Extraction of 3 types of features maps : Color, Itensity, Orientation
2. Center-surround diffenres and normalisation
3. Across-scale combinations and normalisation
4. Linear combinations of the Conspicuity maps obtained at step 3)
5. Final Saliency map


General architecture of the model :

![Model_architecture](https://github.com/ThibBac/Saliency_map/blob/master/images/architecture.png)


#### Specifications of my code

In order to bound the most Salients zones in the image, i used a little trick:

1. First, I take the maximum of the Saliency map
2. I create a bounding box with a fixed size around this zone
3. Then, I hide this zone with a black box
4. Do again the 3 previous steps on the new map with the black box
5. Do all of this steps N times if you want the N most salients objects in the image


#### Experimental results


#### Limitations and potential improvements




## FAQ

#### How to reproduce?

Dowload the Saliency_map.ipynb.

Then change thoses lines according to what you want :
 
```python
train_dir = "data/train_input/resnet_features"
test_dir = "data/test_input/resnet_features"

train_output_filename = "data/train_output.csv"
test_output_filename = "data/test_output.csv"

NB_RUNS = 5               # Number of models to train
model_choice = 'CHOWDER'  # Chose your model : CHOWDER or CHOWDER_test
REG = 0.6                 # Regularization strenght
NB_FOLDS = 3              # Number of crossvalidation's folds
```

## References

**A Model of Saliency-based Visual Attention for Rapid Scene Analysis** 

Laurent Itti, Christof Koch and Ernst Niebur
IEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 20, No. 11, pp. 1254-1259
Nov 1998
