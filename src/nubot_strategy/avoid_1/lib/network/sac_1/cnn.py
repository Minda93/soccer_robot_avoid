#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np
import math
import json

""" keras utils """
from keras.utils import plot_model

""" keras model """
from keras.models import load_model
from keras.models import Model

""" keras layers """
from keras.layers import Dense,Input,Reshape
from keras.layers import Flatten,Activation,Lambda,LeakyReLU
from keras.layers import Concatenate,concatenate

from keras.layers import MaxPooling1D,AveragePooling1D,Conv1D
from keras.layers import BatchNormalization
from keras.layers import Add

import keras.backend as K

""" tensorflow """
import tensorflow as tf

r""" 
    reference
    paper: From Perception to Decision: A Data-driven Approach 
           to End-to-end Motion Planning for Autonomous Ground Robots
"""
# pooling_padding = 'valid' or 'same'

def CNN_Laser_One(input_data,pooling_padding = 'same'):

    """ sh1 """
    sh_1 = Conv1D(filters = 64,\
                    kernel_size = 7,\
                    strides = 3,\
                    padding = 'same')(input_data)
    sh_1 = BatchNormalization()(sh_1)
    sh_1 = Activation('relu')(sh_1)

    sh_1 = MaxPooling1D(3,padding = pooling_padding)(sh_1)
    
    """ sh2 """
    sh_2 = Conv1D(filters = 64,\
                    kernel_size = 3,\
                    strides = 1,\
                    padding = 'same')(sh_1)
    sh_2 = BatchNormalization()(sh_2)
    sh_2 = Activation('relu')(sh_2)

    """ sh3 """
    sh_3 = Conv1D(filters = 64,\
                    kernel_size = 3,\
                    strides = 1,\
                    padding = 'same')(sh_2)
    sh_3 = BatchNormalization()(sh_3)

    """ merge relu 1 """
    add_1 = Add()([sh_1,sh_3])
    add_1 = Activation('relu')(add_1)

    """ sh4 """
    sh_4 = Conv1D(filters = 64,\
                    kernel_size = 3,\
                    strides = 1,\
                    padding = 'same')(add_1)
    sh_4 = BatchNormalization()(sh_4)
    sh_4 = Activation('relu')(sh_4)

    """ sh5 """
    sh_5 = Conv1D(filters = 64,\
                    kernel_size = 3,\
                    strides = 1,\
                    padding = 'same')(sh_4)
    sh_5 = BatchNormalization()(sh_5)

    """ merge relu 2 """
    add_2 = Add()([sh_3,sh_5])
    add_2 = Activation('relu')(add_2)

    """ out """
    out = AveragePooling1D(3,padding = pooling_padding)(add_2)

    out = Flatten()(out)

    return out


"""  """
def CNN_Laser_Two(input_data,pooling_padding = 'same'):
    """ sh1 """
    sh_1 = Conv1D(filters = 32,\
                    kernel_size = 3,\
                    strides = 1,\
                    padding = 'same')(input_data)
    sh_1 = BatchNormalization()(sh_1)
    sh_1 = Activation('relu')(sh_1)

    """ sh2 """
    sh_2 = Conv1D(filters = 64,\
                    kernel_size = 5,\
                    strides = 2,\
                    padding = 'same')(sh_1)
    sh_2 = BatchNormalization()(sh_2)
    sh_2 = Activation('relu')(sh_2)

    """ sh2 """
    sh_3 = Conv1D(filters = 64,\
                    kernel_size = 5,\
                    strides = 4,\
                    padding = 'same')(sh_2)
    sh_3 = BatchNormalization()(sh_3)
    sh_3 = Activation('relu')(sh_3)

    """ out """
    out = Flatten()(sh_3)

    return out

def CNN_Laser_Three(input_data,pooling_padding = 'same'):
    """ sh1 """
    sh_1 = Conv1D(filters = 64,\
                  kernel_size = 7,\
                  strides = 2,\
                  padding = 'same')(input_data)
    sh_1 = BatchNormalization()(sh_1)
    sh_1 = Activation('relu')(sh_1)

    sh_2 = Conv1D(filters = 128,\
                  kernel_size = 5,\
                  strides = 2,\
                  padding = 'same')(sh_1)
    sh_2 = BatchNormalization()(sh_2)
    sh_2 = Activation('relu')(sh_2)

    sh_3 = Conv1D(filters = 256,\
                  kernel_size = 5,\
                  strides = 2,\
                  padding = 'same')(sh_2)
    sh_3 = BatchNormalization()(sh_3)
    sh_3 = Activation('relu')(sh_3)

    sh_4 = Conv1D(filters = 256,\
                  kernel_size = 3,\
                  strides = 1,\
                  padding = 'same')(sh_3)
    sh_4 = BatchNormalization()(sh_4)
    sh_4 = Activation('relu')(sh_4)

    sh_5 = Conv1D(filters = 512,\
                  kernel_size = 3,\
                  strides = 2,\
                  padding = 'same')(sh_4)
    sh_5 = BatchNormalization()(sh_5)
    sh_5 = Activation('relu')(sh_5)

    sh_6 = Conv1D(filters = 512,\
                  kernel_size = 3,\
                  strides = 1,\
                  padding = 'same')(sh_5)
    sh_6 = BatchNormalization()(sh_6)
    sh_6 = Activation('relu')(sh_6)

    sh_7 = Conv1D(filters = 512,\
                  kernel_size = 3,\
                  strides = 2,\
                  padding = 'same')(sh_6)
    sh_7 = BatchNormalization()(sh_7)
    sh_7 = Activation('relu')(sh_7)

    sh_8 = Conv1D(filters = 512,\
                  kernel_size = 3,\
                  strides = 1,\
                  padding = 'same')(sh_7)
    sh_8 = BatchNormalization()(sh_8)
    sh_8 = Activation('relu')(sh_8)

    sh_9 = Conv1D(filters = 1024,\
                  kernel_size = 3,\
                  strides = 2,\
                  padding = 'same')(sh_8)
    sh_9 = BatchNormalization()(sh_9)
    sh_9 = Activation('relu')(sh_9)

    """ out """
    out = Flatten()(sh_9)

    return out