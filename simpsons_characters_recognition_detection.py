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

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json

from skimage.transform import resize
from skimage import data

import warnings
warnings.filterwarnings('ignore')


#==============================================================================
#CLASS
#==============================================================================



#==============================================================================
#FUNCTIONS
#==============================================================================

def load_train_set(dirname,dict_characters):
    """load train data"""
    X_train = []
    Y_train = []
    for label, character in dict_characters.items():
        list_images = os.listdir(dirname+'/'+character)
        for image_name in list_images:
            image =  plt.imread(dirname+'/'+character+'/'+image_name)            
            X_train.append(resize(image,(64,64)))
            Y_train.append(label)
    return np.array(X_train), np.array(Y_train)


def load_test_set(dirname,dict_characters):
    """load test data"""
    X_test = []
    Y_test = []
    for image_name in os.listdir(dirname):
        character_name = "_".join(image_name.split('_')[:-1])
        label = [label for label,character in dict_characters.items() if character == character_name][0]
        image = plt.imread(dirname+'/'+image_name)
        X_test.append(resize(image,(64,64)))
        Y_test.append(label)
    return np.array(X_test), np.array(Y_test)

#==============================================================================
#MAIN
#==============================================================================

#https://www.kaggle.com/alexattia/the-simpsons-characters-dataset

## prepare data

## name of characters
dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11:'lisa_simpson',
        12: 'marge_simpson', 13: 'mayor_quimby',14:'milhouse_van_houten', 15: 'moe_szyslak', 
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}
## load train data
X_train, Y_train = load_train_set("data/simpsons_characters_recognition_detection/the-simpsons-characters-dataset/simpsons_dataset/", dict_characters)       
## load test data
X_test, Y_test = load_test_set("data/simpsons_characters_recognition_detection/the-simpsons-characters-dataset/kaggle_simpson_testset/", dict_characters)
## Scale data
X_train = X_train / 255.0
X_test = X_test / 255.0