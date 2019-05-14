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
                                  units = 512,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_1,\
                                  units = 512,\
                                  activation = tf.nn.relu)
            h_2 = tf.layers.dense(inputs = h_2,\
                                  units = 512,\
                                  activation = tf.nn.relu)

            """ out """
            # mu = tf.layers.dense(inputs = h_2,\
            #                      units = self.action_dim)

            # log_std = tf.layers.dense(h_2, self.action_dim, tf.tanh)
            # log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            
            # std = tf.exp(log_std)

            # pi = mu + tf.random_normal(tf.shape(mu)) * std

            # pre_sum = -0.5 * (((pi - mu) / (tf.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            # logp_pi = tf.reduce_sum(pre_sum, axis=1)

            # mu = tf.tanh(mu)
            # pi = tf.tanh(pi)

            # clip_pi = 1 - tf.square(pi)
            # clip_up = tf.cast(clip_pi > 1, tf.float32)
            # clip_low = tf.cast(clip_pi < 0, tf.float32)
            # clip_pi = clip_pi + tf.stop_gradient((1 - clip_pi) * clip_up + (0 - clip_pi) * clip_low)

            # logp_pi -= tf.reduce_sum(tf.log(clip_pi + self.epsilon), axis=1)

            """ method 2 """
            mu = tf.layers.dense(inputs = h_2,\
                                 units = self.action_dim)
            log_std = tf.layers.dense(inputs = h_2,\
                                 units = self.action_dim)
            log_std = tf.clip_by_value(log_std,self.log_std_min,self.log_std_max)

            std = tf.exp(log_std)
            normal = tf.distributions.Normal(mu,std)
            
            x_t = normal.sample()

            mu = tf.tanh(mu)
            pi = tf.tanh(x_t)

            logp_pi = normal.log_prob(x_t)
            logp_pi -= tf.log(1 - tf.square(pi) + self.epsilon)
            logp_pi = tf.reduce_sum(logp_pi, axis=1)

            return mu, pi, logp_pi
    
    def Eualuate(self,state):
        mu, pi, logp_pi = self.Forward(state)

        mu = tf.multiply(mu,self.action_bound)
        pi = tf.multiply(pi,self.action_bound)

        return mu, pi, logp_pi