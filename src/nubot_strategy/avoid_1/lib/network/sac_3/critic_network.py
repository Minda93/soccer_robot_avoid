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
            """ input """
            state_ = tf.concat([pos, scan_feature], axis=-1)
            state_m = tf.concat([state_, action], axis=-1)
            
            """ layer """
            h_1 = tf.layers.dense(inputs = state_m,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h_2 = tf.layers.dense(inputs = h_1,\
                                  units = 512,\
                                  activation = tf.nn.relu6)
            h_2 = tf.layers.dense(inputs = h_2,\
                                  units = 256,\
                                  activation = tf.nn.relu6)
            
            """ out """
            q_value = tf.layers.dense(inputs = h_2,\
                                      units = 1)
            q_value = tf.squeeze(q_value,axis = 1)              

            return q_value
    
    def Get_Q_Value(self,state,action,reuse = False):
        q_value = self.Forward(state,action,reuse)
        return q_value



