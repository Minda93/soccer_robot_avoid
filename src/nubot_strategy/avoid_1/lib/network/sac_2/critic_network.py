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
class CriticNetwork(object):
    def __init__(self,name,train):
        self.name = name
        self.train = train
    
    def Forward(self, state, action, reuse):
        with tf.variable_scope(self.name,reuse = reuse):
            """ state """
            pos = tf.slice(state,[0,0],[-1,7])
            scan = tf.slice(state,[0,7],[-1,-1])

            scan_feature = CNN_1(scan,self.train)
            state_ = tf.concat([pos, scan_feature], axis=-1)

            """ input """
            state_m = tf.concat([state_, action], axis=-1)
            
            """ layer """
            h_1 = tf.layers.dense(inputs = state_m,\
                                  units = 1024,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_1,\
                                  units = 512,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_2,\
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



