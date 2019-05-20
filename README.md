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

## What is a weakly-supervised approach?

We are talking here of "weakly-supervised approach" beacause our model will run with only the image-level labels or in other terms, without pixel-level annotation. 
This is the case beacause histology slides can be very big (100 000 * 100 000 pixels), and the goal of the algorithm is to classify and localize the diseases on this hudge amount of information.

## CHOWDER Algorithm

## Experiments

#### Data

For this experiment i will use the provided numpy array of features provided by Owkin. 
Each array represent a patient, and each row are the tiles' features exctracted by a ResNet-50 architecure trained on the ImageNet natural image dataset.
We are using the pre-output values of the network with a set of 2048 floatings points.

Each array have a maximum of 1000 tiles but they but there may be less.

#### Network Architecture

The data is processing using thoses steps :

1. Load the provided numpy arrays using np.load
2. If there is less than 1000 tiles in it, complete it with 0s
3. Add the resulting array to a feature array.
4. Do it for the train and test data.


The CHOWDER architecture is defined as follow :
![CHOWDER architecture](https://github.com/ThibBac/Ownkin-s_Homework/blob/master/images/architecture.PNG)

The data provided correspond to the "Local Descpitor", the expirement is the following steps.

The first layer is an one dimensional convolution layer which will be a feature embedding layer. It will take the input data of size (1000, 2048) and for each tile provide a score according to the potentiality that this tile is cancerous or not. The output is a feature vector of size (1000, 1).

We then take the R max and min instances of this vector which are then concatenated, the output is a 2R vector.

Finaly this vector is fed into a MLP network composed of 3 Dense layers of size 200, 100 and 1 for the output, with sigmoids activations.
The final output is a floating number corresponding to probability of the "cancerousness" of the cells in the image.

An l2-regularization of 0.5 is using on the first layer and dropout of 0.5 between the Dense layers of the MLP.


#### Training

At first, I initialize all weight matrices randomly using Le_Cun normal distribution with a random seed.

The model is optimized using ADAM on a binary crossentropy with a batch size of 10 during 30 epochs.
The learning rate is set to 0.001 and i am using keras' ReduceLROnPlateau callback to decrease it if the learning start to stagnate.

The training is done using a K-fold cross validation with a number of folds defined to 3.

In order to reduce variance and over-fitting, i'am using an ensembling method which will average the output of multiple models which differs only by their initial weights.



#### Specifications of my code

After many tests, i found out that the model have some difficulties to converge. So i try to tune the learning rate when the accuracy of the model is not increasing anymore.
I used Keras backend, ReduceLROnPlateau which will deacrease the lr to 90% of his value if the accuracy stagnated for 3 epochs.

I also found that the final result was closely linked to the initalisation of the weight of the network. So i try different initializers which are RandomUniform, TruncatedNormal and lecun_normal. 
The initializer that gave me the best results was Keras Lecun_normal.


I also try to implement an architecture slightly different of the CHOWDER algorithm.
With the idea that max instances are as important as min instances, i implemented two differents MLP classifier, one for the top and one for the min instances.
I then averaging the outputs of this two classifier, together.
This network gave me about the same results as the CHOWDER algorithm.


#### Experimental results

The metric used to judge the model efficiency is the AUC score.

Those curves were generated by training 5 differents model and ensembling them.

This first result is obtained with the standard implementation of the CHOWDER algorithm.
We can see that it is slightly overfitting.

![CHOWDER results](https://github.com/ThibBac/Ownkin-s_Homework/blob/master/images/results1.PNG)


This second result is obtained with my custom CHOWDER algorithm, with 2 diffents classifier.
Those results are very similar to the original implementation.

![CHOWDERtest results](https://github.com/ThibBac/Ownkin-s_Homework/blob/master/images/results2.PNG)


#### Limitations and potential improvements
  
1. The dataset is small so the network can have difficulties to converge and generalize well.
With acces to the initial image, we can generate more tiles using ImageDataGenerator, indeed a tiles that is flipped or rotated is still cancerous or healthy.

2. Greats improvemnents can be done by a better tuning of the hyper-parameter, i talk about it in the next section.



#### Bonus and open question: Basic AutoML

There is plenty of hyper-parameters on this algorithm so it's primordial to correctly tune them.
Here i tried the grid search to have good view of the best hyper-parameters.
It's simple to implement :
```python
initilizers = [initializers.lecun_normal(seed=12), initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=12), initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=12)]
weight_decays = [0.7, 0.5, 0.2]
batch_sizes  = [10, 20, 30]

for ini in initilizers:
  for weight_decay in weight_decays:
    for batch_size in batch_sizes:
```
But this method is very expensive in term of computer power. 

An other method is the random search, which is pretty similar but instead of having defined value we can use random functions like 
np.random.uniform from numpy.

This illustration represents well how random search is working:

![Random_search](https://github.com/ThibBac/Ownkin-s_Homework/blob/master/images/Random_search.PNG)

From this paper : http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf


The last method i want to talk about is the Gaussian process. This method will uses a set of previously evaluated parameters and resulting accuracy to make an assumption about unobserved parameters.

I try to implement it but too late to have results. But i found this article pretty interesting :

http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html?fbclid=IwAR3zneC32wWJyqt1i3jNnkxXHb8zaYOfcpm6a5YD0a83Lo9cyAAQq-4LJKU#bayesian-optimization


## FAQ

#### How to reproduce?

Dowload the dataset and the CHOWDER.ipynb.

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
