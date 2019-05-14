#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np
import math
import json

""" tensorflow """
import tensorflow as tf

""" one state """
class ValueNetwork(object):
    def __init__(self,name):
        self.name = name
        
    def Forward(self,state):
        with tf.variable_scope(self.name):
            """ state """
            pos = tf.slice(state,[0,0],[-1,7])
            scan = tf.slice(state,[0,7],[-1,-1])

            """ scan """
            scan_feature = tf.layers.dense(inputs = scan,\
                                           units = 512,\
                                           activation = tf.nn.relu6)
            scan_feature = tf.layers.dense(inputs = scan_feature,\
                                           units = 256,\
                                           activation = tf.nn.relu6)
            scan_feature = tf.layers.dense(inputs = scan_feature,\
                                           units = 128,\
                                           activation = tf.nn.relu6)
            """ input """
            state_ = tf.concat([pos, scan_feature], axis=-1)

            """ layer """
            h_1 = tf.layers.dense(inputs = state_,\
                                  units = 256,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_1,\
                                  units = 256,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_2,\
                                  units = 256,\
                                  activation = tf.nn.relu)

            """ out """
            value = tf.layers.dense(inputs = h_2,\
                                    units = 1)
            value = tf.squeeze(value,axis = 1)

            return value

    def Get_Value(self,state):
        value = self.Forward(state)
        return value



