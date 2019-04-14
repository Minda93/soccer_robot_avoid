#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" ros """
import roslib
roslib.load_manifest('nubot_strategy')
import rospy
import rospkg

""" lib """
# policy
from lib.network.sac.sac import SAC

from lib.tool.utils import ReplayBuffer_Deque
from lib.env.avoid_env_soccer import NubotAvoidEnv

""" tool """
import math
import os
import numpy as np
import pickle
import threading

PACKAGE_PATH = rospkg.RosPack().get_path('nubot_strategy')+'/avoid/'


class PlayGame(object):
    def __init__(self):
        self.Init_Param()

        """ env """
        self.env = NubotAvoidEnv(robot_name = 'nubot1',\
                                 state_dim = self.state_dim,\
                                 action_dim = self.action_dim,
                                 seed = self.seed)
        """ set seed """
        np.random.seed(self.seed)

        """ init policy """
        if(self.policy_name == "sac"):
            self.policy = SAC(np.sum(self.state_dim),self.action_dim,self.action_bound[1],self.save_path)
            if(self.load_models):
                self.policy.Load(self.load_path,self.load_file_name)
        
        """ init replay buffer """
        self.replay_buffer = ReplayBuffer_Deque(self.buffer_size)
        if(self.load_buffer):
            self.Load_Replay_Buffer(self.load_buffer_episode)
    
    def Init_Param(self):
        # policy name
        self.policy_name = "sac"

        # Set env and numpy seeds
        self.seed = 0

        # buffer size
        self.buffer_size = 1000000

        # How many time steps purely random policy is run for
        self.start_timesteps = 10000
        # self.start_timesteps = 0

        # log interval
        self.log_interval = 10

        # good job 
        self.good_job = 3500.

        # How often (time steps) we evaluate
        self.eval_freq = 200

        # Max time steps to run environment for
        self.max_timesteps = 2000

        # Max episode to run environment for
        self.max_episodes = 5000
        # self.max_episodes = 200

        # How many time episodes purely random robot is run for
        # self.start_episodes = int(self.max_episodes/2)
        self.start_episodes = 3000

        # Std of Gausssian exploration noise
        # self.expl_noise = 0.1
        self.expl_noise = 0

        # whether or not models are trained
        self.train = False

        # whether or not models are saved
        self.save_models = True

        # whether or not buffer are Loaded
        self.load_buffer = False
        self.load_buffer_episode = 3000
        
        # whether or not models are Loaded
        self.load_models = True

        # random
        self.random_robot = True
        self.random_goal = True

        # load file_name
        load_model = 'model_1'
        seed = 0
        load_episode = 3000
        self.load_path =  PACKAGE_PATH+"config/{}/{}/{}/".format(self.policy_name,load_model,load_episode)
        self.load_file_name = "{}_{}".format(self.policy_name,seed)
        self.load_buffer_path = "./config/{}/{}/".format(self.policy_name,load_model)

        # save file_name
        output_model = "model_1"
        self.save_path = PACKAGE_PATH+"config/{}/{}/".format(self.policy_name,output_model)
        self.save_file_name = "{}_{}".format(self.policy_name,self.seed)
        self.check_save_path = self.Check_Path(self.save_path)

        """ init game """
        self.state_dim = [7,360]
        self.action_dim = 1
        self.action_bound = [-math.pi,math.pi]
    
    def Play(self):
        total_timesteps = 0
        episode_tmp = 0
        timesteps_since_eval = 0

        avg_reward = 0
        episode_reward = 0

        """ training procedure """
        for episode in range(1,self.max_episodes+1):
            if(self.random_goal and self.random_robot and episode < self.start_episodes):
                state = self.env.Reset(3)
                print('run episode {} '.format(episode),end="")
            elif(self.random_robot and episode < self.start_episodes):
                state = self.env.Reset(2)
                print('run episode {} '.format(episode))
            else:
                state = self.env.Reset()
                print('run episode {} '.format(episode))

            """ 30hz """
            rate = rospy.Rate(20)
            for step in range(1,self.max_timesteps+1):
                total_timesteps += 1
                # print(total_timesteps)
                action = self.policy.Select_Action(state,self.train)

                # print(action,end="")
                if(self.expl_noise != 0):
                    action = (action + np.random.normal(0,self.expl_noise,size=self.action_dim)).clip(*self.action_bound)
                # print(action)

                new_state,reward,done,info = self.env.Step(action)

                avg_reward += reward
                episode_reward += reward

                """ store """
                self.replay_buffer.Add(state,\
                                       np.reshape(action,(self.action_dim,)),\
                                       reward,\
                                       new_state,\
                                       done)
                state = new_state
                timesteps_since_eval += 1

                # if(total_timesteps > self.start_timesteps):
                #     self.policy.Train(self.replay_buffer,total_timesteps-self.start_timesteps)

                if(done or step == (self.max_timesteps - 1)):
                    self.env.Stop_Env()
                    if(self.train):
                        if(total_timesteps > self.start_timesteps):
                            loss = self.policy.Train(self.replay_buffer,step,episode-episode_tmp)
                        else:
                            episode_tmp += 1
                    break
                
                rate.sleep()

            """ save good job """
            if(self.train):
                self.Log(episode, episode_reward, step,info)
                if(episode % self.log_interval == 0 and total_timesteps > self.start_timesteps):
                    if(avg_reward/self.log_interval >= self.good_job):
                        print("#########################")
                        print("success episode ", episode)
                        print("save model")
                        print("#########################")
                        self.Save(episode)

                if(episode % self.eval_freq == 0):
                    self.Save(episode)

                if(episode % 1000 == 0):
                    self.Save_Replay_Buffer(episode)
            else:
                self.Log_Show(episode, episode_reward, step, info)
            
            episode_reward = 0

            if(episode % self.log_interval == 0):
                avg_reward /= self.log_interval
                print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
                avg_reward = 0

        if(self.train):   
            self.Save(episode)
            self.Save_Replay_Buffer(episode)
        else:
            # self.Save_Replay_Buffer(episode)
            pass
        
    """ tool """
    def Save(self,episode):
        if(self.check_save_path):
            path = self.save_path  + str(episode) + '/'
            
            self.Check_Path(path)
            self.policy.Save(path,self.save_file_name)
            print("finish save model",episode)
        else:
            print("not find path")
    
    def Save_Replay_Buffer(self,episode):
        if(self.check_save_path):
            with open(self.save_path+"rb_{}.pickle".format(episode),'wb') as f:
                pickle.dump(self.replay_buffer.buffer,f)
        else:
            print("not find path")
    
    def Load_Replay_Buffer(self,episode):
        with open(self.load_buffer_path+"rb_{}.pickle".format(episode),'rb') as f:
            data = pickle.load(f)
            self.replay_buffer.buffer = data
    
    def Log(self,episode, episode_reward, step,info):
        if(self.check_save_path):
            with open(self.save_path+'log.txt','a') as f:
                f.write("{},{},{},{}\n".format(episode,episode_reward,step,info))
        else:
            print("not find path")
    
    def Log_Show(self,episode, episode_reward, step,info):
        if(self.check_save_path):
            with open(self.save_path+'log_show.txt','a') as f:
                f.write("{},{},{},{}\n".format(episode,episode_reward,step,info))
        else:
            print("not find path")
    
    def Check_Path(self,path):
        if (not os.path.exists(path)):
            os.makedirs(path)
        return True

def main():
    rospy.init_node('nubot_avoid', anonymous=True)
    game = PlayGame()

    game.Play()


if __name__ == '__main__':
    main()
