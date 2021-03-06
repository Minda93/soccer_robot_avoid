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
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.state_dim = np.sum(state_dim)
        self.robot_dim = state_dim[0]
        self.scan_dim = state_dim[1]

        self.Init_Param()

        """ model """
        self.actor = ActorNetwork('Actor',action_dim,action_bound,self.robot_dim,self.scan_dim)
        self.critic_Q1 = CriticNetwork('Q_value1',self.robot_dim,self.scan_dim)
        self.critic_Q2 = CriticNetwork('Q_value2',self.robot_dim,self.scan_dim)
        self.value_net = ValueNetwork('Value',self.robot_dim,self.scan_dim)
        self.target_value_net = ValueNetwork('Target_Value',self.robot_dim,self.scan_dim)

        self.Build_Model()

        """ limit GPU"""
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)

        """ save """
        self.saver = tf.train.Saver()

        """ tensorflow """
        self.writer = tf.summary.FileWriter(logpath+"/TensorBoard/",graph = self.sess.graph)

    def Init_Param(self):
        """ Target Network HyperParameters """
        self.TAU = 0.995

        """ discount """
        self.gamma = 0.99
        self.alpha = 0.2
        self.auto_alpha = True

        """ learning rate """
        self.lr_a = 0.0003
        self.lr_v = 0.0003
        
        """ batch size """
        # self.batch_size = 100
        self.batch_size = 256
    
    def Build_Model(self):
        """ input """
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name="state")
        self.new_state = tf.placeholder(tf.float32, [None, self.state_dim], name="new_state")
        self.action = tf.placeholder(tf.float32, [None, self.action_dim], name="action")
        self.reward = tf.placeholder(tf.float32, [None,], name="reward")
        self.done = tf.placeholder(tf.float32, [None,], name="done")

        """ episode reward """
        self.r_t = tf.placeholder(tf.float32, name="r_t")

        """ alpha """
        if(self.auto_alpha):
            self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,\
                                                initializer=np.log(self.alpha).astype(np.float32))
            self.ent_coef = tf.exp(self.log_ent_coef)
            self.alpha = self.ent_coef

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
        q1_value_loss = 0.5*tf.reduce_mean(tf.squared_difference(y_q, q1_value))
        q2_value_loss = 0.5*tf.reduce_mean(tf.squared_difference(y_q, q2_value))
        value_loss = 0.5*tf.reduce_mean(tf.squared_difference(y_v, value))
        total_value_loss = q1_value_loss + q2_value_loss + value_loss

        if(self.auto_alpha):
            self.target_entropy = -np.prod(self.action_dim).astype(np.float32)
            ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
        
        loss_actor = tf.summary.scalar("loss_actor",actor_loss)
        loss_q1 = tf.summary.scalar("loss_q1",q1_value_loss)
        loss_q2 = tf.summary.scalar("loss_q2",q2_value_loss)
        loss_value = tf.summary.scalar("loss_value",value_loss)
        loss_t_value = tf.summary.scalar("loss_t_value",total_value_loss)

        
        if(self.auto_alpha):
            loss_ent_coef = tf.summary.scalar("loss_ent_coef",ent_coef_loss)
            self.merge_summary = tf.summary.merge([loss_actor,\
                                               loss_q1,\
                                               loss_q2,\
                                               loss_value,\
                                               loss_t_value,\
                                               loss_ent_coef])
        else:
            self.merge_summary = tf.summary.merge([loss_actor,\
                                               loss_q1,\
                                               loss_q2,\
                                               loss_value,\
                                               loss_t_value])

        episode_reward = tf.summary.scalar("episode_reward",self.r_t)

        self.reward_summary = tf.summary.merge([episode_reward])
        
        """ train """
        actor_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_a)
        actor_train_op = actor_optimizer.minimize(actor_loss,var_list = tf.global_variables('Actor'))
        value_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_v)
        value_params = tf.global_variables('Q_value1') + tf.global_variables('Q_value2') + tf.global_variables('Value')
        if(self.auto_alpha):
            entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_a)

        """ method 1 """
        # with tf.control_dependencies([actor_train_op]):
        #     value_train_op = value_optimizer.minimize(total_value_loss, var_list = value_params)
        with tf.control_dependencies([actor_train_op]):
            q1value_train_op = value_optimizer.minimize(total_value_loss, var_list = tf.global_variables('Q_value1'))
        with tf.control_dependencies([q1value_train_op]):
            q2value_train_op = value_optimizer.minimize(total_value_loss, var_list = tf.global_variables('Q_value2'))
        with tf.control_dependencies([q2value_train_op]):
            value_train_op = value_optimizer.minimize(total_value_loss, var_list = tf.global_variables('Value'))

        if(self.auto_alpha):
            with tf.control_dependencies([value_train_op]):
                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
            with tf.control_dependencies([ent_coef_op]):
                self.target_update = [tf.assign(tv, self.TAU * tv + (1 - self.TAU) * v)\
                                    for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]
        else:
            with tf.control_dependencies([value_train_op]):
                self.target_update = [tf.assign(tv, self.TAU * tv + (1 - self.TAU) * v)\
                                    for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        """ method 2 """
        # q_value_train_op = value_optimizer.minimize(q1_value_loss, var_list = tf.global_variables('Q_value1'))
        # with tf.control_dependencies([q_value_train_op]):
        #     self.q_value_update = value_optimizer.minimize(q2_value_loss, var_list = tf.global_variables('Q_value2'))

        # if(self.auto_alpha):
        #     with tf.control_dependencies([actor_train_op]):
        #         value_coef_op = value_optimizer.minimize(value_loss, var_list=tf.global_variables('Value'))
        #     with tf.control_dependencies([value_coef_op]):
        #         ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
        #     with tf.control_dependencies([ent_coef_op]):
        #         self.target_update = [tf.assign(tv, self.TAU * tv + (1 - self.TAU) * v)\
        #                             for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]
        # else:
        #     with tf.control_dependencies([actor_train_op]):
        #         value_coef_op = value_optimizer.minimize(value_loss, var_list=tf.global_variables('Value'))
        #     with tf.control_dependencies([value_coef_op]):
        #         self.target_update = [tf.assign(tv, self.TAU * tv + (1 - self.TAU) * v)\
        #                             for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

        self.target_init = [tf.assign(tv, v)\
                            for v, tv in zip(tf.global_variables('Value'), tf.global_variables('Target_Value'))]

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
    
    def Train(self,replay_buffer,iterations,episode = 0,gradient_steps = 1,delay = 2):
        if(episode == 0):
            """ sample """
            for i in range(gradient_steps):
                state, action, reward, new_state, done = replay_buffer.Sample(self.batch_size)

                feed_dict = {self.state: state,\
                                self.action: action,\
                                self.new_state: new_state,\
                                self.reward: reward,\
                                self.done: np.float32(done)}

                # self.sess.run(self.q_value_update, feed_dict)

                if(iterations % delay == 0):
                    self.sess.run(self.target_update, feed_dict)

            train_summary = self.sess.run(self.merge_summary, feed_dict)
            self.writer.add_summary(train_summary,iterations)
        else:
            for it in range(iterations*gradient_steps):
                """ sample """
                state, action, reward, new_state, done = replay_buffer.Sample(self.batch_size)

                feed_dict = {self.state: state,\
                            self.action: action,\
                            self.new_state: new_state,\
                            self.reward: reward,\
                            self.done: np.float32(done)}
                
                # self.sess.run(self.q_value_update, feed_dict)

                if(it % delay == 0):
                    self.sess.run(self.target_update, feed_dict)

            train_summary = self.sess.run(self.merge_summary, feed_dict)
            self.writer.add_summary(train_summary,episode)
        
    def Log(self,reward,episode):
        feed_dict = {self.r_t: reward}
        reward_summary = self.sess.run(self.reward_summary, feed_dict)
        self.writer.add_summary(reward_summary,episode)
    
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


