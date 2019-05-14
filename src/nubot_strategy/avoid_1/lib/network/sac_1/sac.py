#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" keras """
import keras.backend as K

""" tensorflow """
import tensorflow as tf

""" tool """
import numpy as np
import math

""" lib """
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from .value_network import ValueNetwork
from lib.tool.utils import ReplayBuffer_Deque

class SAC(object):
    def __init__(self,state_dim,action_dim,action_bound,logpath):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.Init_Param()

        """ model """
        self.actor = ActorNetwork('Actor',action_dim,action_bound)
        self.critic_Q1 = CriticNetwork('Q_value1')
        self.critic_Q2 = CriticNetwork('Q_value2')
        self.value_net = ValueNetwork('Value')
        self.target_value_net = ValueNetwork('Target_Value')

        self.Build_Model()

        """ save """
        self.saver = tf.train.Saver()

        self.merge_summary = tf.summary.merge_all()
        """ tensorflow """
        self.writer = tf.summary.FileWriter(logpath+"/TensorBoard/",graph = self.sess.graph)

    def Init_Param(self):
        """ Target Network HyperParameters """
        self.TAU = 0.995

        """ discount """
        self.gamma = 0.99
        self.alpha = 0.5

        """ learning rate """
        self.lr_a = 0.001
        self.lr_v = 0.001
        # self.lr_a = 0.003
        # self.lr_v = 0.003
        
        """ batch size """
        # self.batch_size = 128
        self.batch_size = 100
    
    def Build_Model(self):
        """ input """
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name="state")
        self.new_state = tf.placeholder(tf.float32, [None, self.state_dim], name="new_state")
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], name="action")
        self.reward = tf.placeholder(tf.float32, [None,], name="reward")
        self.done = tf.placeholder(tf.float32, [None,], name="done")

        self.mu, self.pi, logp_pi = self.actor.Eualuate(self.state)

        """ get value """
        q1_value = self.critic_Q1.Get_Q_Value(self.state,self.action)
        q2_value = self.critic_Q2.Get_Q_Value(self.state,self.action)

        q1_value_pi = self.critic_Q1.Get_Q_Value(self.state,self.pi,reuse=True)
        q2_value_pi = self.critic_Q2.Get_Q_Value(self.state,self.pi,reuse=True)
        min_q_value = tf.minimum(q1_value_pi,q2_value_pi)

        value = self.value_net.Get_Value(self.state)
        target_value = self.target_value_net.Get_Value(self.new_state)

        """ compute target for Q and Value """
        y_q = tf.stop_gradient(self.reward + self.gamma*(1 - self.done)*target_value)
        y_v = tf.stop_gradient(min_q_value - self.alpha*logp_pi)

        """ loss """
        actor_loss = tf.reduce_mean(self.alpha*logp_pi - q1_value_pi)
        q1_value_loss = tf.reduce_mean(tf.squared_difference(y_q, q1_value))
        q2_value_loss = tf.reduce_mean(tf.squared_difference(y_q, q2_value))
        value_loss = tf.reduce_mean(tf.squared_difference(y_v, value))

        total_value_loss = q1_value_loss + q2_value_loss + value_loss
        
        tf.summary.scalar("loss_actor",actor_loss)
        tf.summary.scalar("loss_q1",q1_value_loss)
        tf.summary.scalar("loss_q2",q2_value_loss)
        tf.summary.scalar("loss_value",value_loss)
        tf.summary.scalar("loss_t_value",total_value_loss)
        
        """ train """
        actor_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_a)
        actor_train_op = actor_optimizer.minimize(actor_loss,var_list = tf.global_variables('Actor'))
        value_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_v)
        value_params = tf.global_variables('Q_value') + tf.global_variables('Value')

        with tf.control_dependencies([actor_train_op]):
            value_train_op = value_optimizer.minimize(total_value_loss, var_list = value_params)
        with tf.control_dependencies([value_train_op]):
            self.target_update = [tf.assign(tv, self.TAU * tv + (1 - self.TAU) * v)\
                                            for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        target_init = [tf.assign(tv, v)\
                       for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def Select_Action(self,state,train):
        if(train):
            action = self.sess.run(self.pi,feed_dict={self.state: state.reshape(1, -1)})
            # action = np.squeeze(action, axis=1)
            action = np.reshape(action,(self.action_dim,))
        else:
            action = self.sess.run(self.mu,feed_dict={self.state: state.reshape(1, -1)})
            # action = np.squeeze(action, axis=1)
            action = np.reshape(action,(self.action_dim,))
        return action
    
    def Train(self,replay_buffer,iterations,episode = 0):
        if(episode == 0):
            """ sample """
            state, action, reward, new_state, done = replay_buffer.Sample(self.batch_size)

            feed_dict = {self.state: state,\
                            self.action: action,\
                            self.new_state: new_state,\
                            self.reward: reward,\
                            self.done: np.float32(done)}

            self.sess.run(self.target_update, feed_dict)

            train_summary = self.sess.run(self.merge_summary, feed_dict)
            self.writer.add_summary(train_summary,iterations)
        else:
            for it in range(iterations):
                """ sample """
                state, action, reward, new_state, done = replay_buffer.Sample(self.batch_size)

                feed_dict = {self.state: state,\
                            self.action: action,\
                            self.new_state: new_state,\
                            self.reward: reward,\
                            self.done: np.float32(done)}

                self.sess.run(self.target_update, feed_dict)

            train_summary = self.sess.run(self.merge_summary, feed_dict)
            self.writer.add_summary(train_summary,episode)
        
    def Log(self):
        pass
    
    def Save(self,directory,filename):
        path = "{}{}_Model.ckpt".format(directory,filename)
        self.saver.save(self.sess, path)
    
    def Load(self,directory,filename):
        checkpoint = tf.train.get_checkpoint_state("{}".format(directory))

        if(checkpoint and checkpoint.model_checkpoint_path):
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


