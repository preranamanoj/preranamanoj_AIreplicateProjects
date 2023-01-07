# -*- coding: utf-8 -*-
"""MNISTFashionProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rrmFxp_sjALWfNUGClpvgk0OyS77O0g-
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Flatten, Dense 
import matplotlib.pyplot as plt
# %matplotlib inline

#adding dataset
from keras.datasets import fashion_mnist 
(x_train, x_lab),(y_test, y_lab) = fashion_mnist.load_data() #train using x dataset and x labels to help the computer #out of the 70k images, 60k is used for training and 10k for testing
plt.imshow(x_train[0]) #showing the index/th image of the x train in the square brackets 
plt.title('Class: {}'.format(x_lab[0])) #just for the labelling on top of the image, e.g: class 9 below on top of index 0 of x train.
plt.figure() #informs you about figure size; not much use other than that

x_train = tf.keras.utils.normalize(x_train, axis = 1)
y_test = tf.keras.utils.normalize(y_test, axis = 1)
plt.imshow(x_train[0])
plt.title('Class: {}'.format(x_lab[0])) #just for the labelling on top of the image, e.g: class 9 below on top of index 0 of x train.
plt.figure() #informs you about figure size; not much use other than that