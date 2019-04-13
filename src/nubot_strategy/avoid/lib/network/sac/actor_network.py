#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" tool """
import numpy as np
import math
import json

""" tensorflow """
import tensorflow as tf

""" one state """
class ActorNetwork(object):
    def __init__(self,name,action_dim,action_bound): 
        self.name = name
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.log_std_min = -20
        self.log_std_max = 2

        self.EPS = 1e-8
        self.epsilon = 1e-6

    def Forward(self,state):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
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
            mu = tf.layers.dense(inputs = h_2,\
                                 units = self.action_dim)

            log_std = tf.layers.dense(h_2, self.action_dim, tf.tanh)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            
            std = tf.exp(log_std)

            pi = mu + tf.random_normal(tf.shape(mu)) * std

            pre_sum = -0.5 * (((pi - mu) / (tf.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            logp_pi = tf.reduce_sum(pre_sum, axis=1)

            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            clip_pi = 1 - tf.square(pi)
            clip_up = tf.cast(clip_pi > 1, tf.float32)
            clip_low = tf.cast(clip_pi < 0, tf.float32)
            clip_pi = clip_pi + tf.stop_gradient((1 - clip_pi) * clip_up + (0 - clip_pi) * clip_low)

            logp_pi -= tf.reduce_sum(tf.log(clip_pi + self.epsilon), axis=1)

            return mu, pi, logp_pi
    
    def Eualuate(self,state):
        mu, pi, logp_pi = self.Forward(state)

        mu *= self.action_bound
        pi *= self.action_bound

        return mu, pi, logp_pi