#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np
import math
import json

""" tensorflow """
import tensorflow as tf

""" lib """
from .cnn import CNN_1

""" one state """
class ValueNetwork(object):
    def __init__(self,name,train):
        self.name = name
        self.train = train
        
    def Forward(self,state):
        with tf.variable_scope(self.name):
            """ state """
            pos = tf.slice(state,[0,0],[-1,7])
            scan = tf.slice(state,[0,7],[-1,-1])

            scan_feature = CNN_1(scan,self.train)
            state_ = tf.concat([pos, scan_feature], axis=-1)

            """ layer """
            h_1 = tf.layers.dense(inputs = state_,\
                                  units = 1024,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_1,\
                                  units = 512,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_2,\
                                  units = 512,\
                                  activation = tf.nn.relu)

            """ out """
            value = tf.layers.dense(inputs = h_2,\
                                    units = 1)
            value = tf.squeeze(value,axis = 1)

            return value

    def Get_Value(self,state):
        value = self.Forward(state)
        return value



