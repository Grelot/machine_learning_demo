#===============================================================================
#INFORMATION
#===============================================================================
# Codes for Predicting Housing Prices:
# A model to predict the value of a given house in the Boston real estate market
# using various statistical analysis tools.
# Identified the best price that a client can sell their house 
# using machine learning.
# 
# Guerin Pierre-Edouard
#
#
# git repository : https://github.com/Grelot/machine_learning_demo
#
# kaggle profile : https://www.kaggle.com/pierreedouardguerin
# 
# kaggle compete : https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#
#==============================================================================
#NOTICE
#==============================================================================
#
#
#
#
#==============================================================================
#MODULES
#==============================================================================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import scipy
import cv2
np.random.seed(2)
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json

import warnings

warnings.filterwarnings('ignore')


#==============================================================================
#CLASS
#==============================================================================



#==============================================================================
#FUNCTIONS
#==============================================================================



#==============================================================================
#MAIN
#==============================================================================

https://www.kaggle.com/alexattia/the-simpsons-characters-dataset
