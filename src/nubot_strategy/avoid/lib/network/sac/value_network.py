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
            """ layer """
            h_1 = tf.layers.dense(inputs = state,\
                                  units = 512,\
                                  activation = tf.nn.relu)
            # h_2 = tf.layers.dense(inputs = h_1,\
            #                       units = 300,\
            #                       activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_1,\
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



