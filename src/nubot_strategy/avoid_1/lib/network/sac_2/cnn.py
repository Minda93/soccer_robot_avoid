#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np

""" tensorflow """
import tensorflow as tf

def CNN_1(state,train):
    
    state_ = tf.reshape(state,[-1,state.shape[1],1])

    conv1 = tf.layers.conv1d(inputs = state_,\
                             filters = 64,\
                             kernel_size = 3,\
                             strides = 2,\
                             padding='valid')
    # conv1 = tf.layers.batch_normalization(conv1, training=train)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.layers.conv1d(inputs = conv1,\
                             filters = 32,\
                             kernel_size = 3,\
                             strides = 2,\
                             padding='valid')
    # conv2 = tf.layers.batch_normalization(conv2, training=train)
    conv2 = tf.nn.relu(conv2)
    
    # conv3 = tf.layers.conv1d(inputs = conv2,\
    #                          filters = 32,\
    #                          kernel_size = 3,\
    #                          strides = 1,\
    #                          padding='valid')
    # conv3 = tf.layers.batch_normalization(conv3, training=train)
    # conv3 = tf.nn.relu(conv3)

    out = tf.layers.flatten(conv2)
    out = tf.layers.dense(inputs = out,\
                          units = 1024,\
                          activation = tf.nn.relu)

    return out


