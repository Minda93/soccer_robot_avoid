#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np
import math
import json

""" tensorflow """
import tensorflow as tf

""" one state """
class CriticNetwork(object):
    def __init__(self,name):
        self.name = name
    
    def Forward(self, state, action, reuse):
        with tf.variable_scope(self.name,reuse = reuse):
            """ input """
            state_m = tf.concat([state, action], axis=-1)
            
            """ layer """
            h_1 = tf.layers.dense(inputs = state_m,\
                                  units = 512,\
                                  activation = tf.nn.relu)
            # h_2 = tf.layers.dense(inputs = h_1,\
            #                       units = 300,\
            #                       activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_1,\
                                  units = 512,\
                                  activation = tf.nn.relu)
            
            """ out """
            q_value = tf.layers.dense(inputs = h_2,\
                                      units = 1)
            q_value = tf.squeeze(q_value,axis = 1)              

            return q_value
    
    def Get_Q_Value(self,state,action,reuse = False):
        q_value = self.Forward(state,action,reuse)
        return q_value



