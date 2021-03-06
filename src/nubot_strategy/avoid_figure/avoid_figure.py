#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" ros """
import roslib
roslib.load_manifest('nubot_strategy')
import rospy
import rospkg
from nubot_strategy.msg import tutor

""" lib """
from lib.tool.utils import ReplayBuffer_Deque
""" tool """
import math
import os
import numpy as np
import pickle
import threading

PACKAGE_PATH = rospkg.RosPack().get_path('nubot_strategy')+'/avoid_figure/'

class NodeHandle(object):
    def __init__(self):
        self.pub_tutor = rospy.Publisher('/tutor',tutor, queue_size = 1)
    
    def Pub_Tutor(self,s,a,r,s1,d):
        buffer = tutor()
        buffer.state = s
        buffer.action = a
        buffer.reward = r
        buffer.new_state = s1
        buffer.done = d

        self.pub_tutor.publish(buffer)

class PlayGame(object):
    def __init__(self):
        self.Init_Param()

        """ define action """
        # self.Define_Action(r_info=5,s_info=360,a_info='angle')
        self.Define_Action(r_info=5,s_info=120,a_info='angle')
        """ Build env """
        self.Build_Env(robot = 'nubot1',choose = 3,a_info='angle',dynamic = False)

        """ set seed """
        np.random.seed(self.seed)

        self.nh = NodeHandle()

        """ Build Policy """
        self.Build_Policy()

        """ init replay buffer """
        self.replay_buffer = ReplayBuffer_Deque(self.buffer_size)
        if(self.load_buffer):
            self.Load_Replay_Buffer(self.load_buffer_episode)

    def Init_Param(self):
        # """ policy name """
        self.policy_name = "sac_5"

        # """ Set env and numpy seeds """
        self.seed = 9

        # """ buffer size """
        self.buffer_size = 1000000

        # """ log interval """
        self.log_interval = 100

        # good job 
        self.good_job = 75

        # How often (time steps) we evaluate
        self.eval_freq = 1000

        # """ How many time steps purely random policy is run for """
        # self.start_timesteps = 10000
        self.start_timesteps = 256*2
    
        # Max time steps to run environment for
        self.max_timesteps = 400
        # self.max_timesteps = 1000

        # Max episode to run environment for
        self.max_episodes = 50000
        # self.max_episodes = 100

        # """ Std of Gausssian exploration noise """
        self.expl_noise = 0

        # """ whether or not models are trained """
        self.train = True

        # """ whether or not models are Loaded """
        self.load_models = True

        # """ whether or not robot's route are saved """
        self.save_route = False

        # """ whether or not buffer are Loaded """
        self.load_buffer = False

        # """ episode update """
        self.episode_update = True

        # """ render """
        self.render = True
        self.tutor = False

        # r""" env reset
        #     reset = 0 -> repeat pos
        #     reset = 1 -> random obstacle  - Challenge goal and robot
        #     reset = 2 -> randorm robot and obstacle - Challenge goal
        #     reset = 3 -> all random
        #     reset = 4 -> random goal  - current robot and current obstacle
        #     reset = 5 -> random robot  -  current obstacle and current goal
        #     reset = 6 -> random goal and robot - current obstacle
        #     reset = 7 -> read train batch - random goal and random robot 
        #               -> need to read batch file
        # """
        self.reset_flag = 3
        self.batch_file = -1

        # """ file processing """
        # Load model
        if(self.load_models):
            load_model = 'model_3'
            seed = 3
            load_episode = 5000
            self.load_path =  PACKAGE_PATH+\
                "config/{}/{}/{}/".format(self.policy_name,load_model,load_episode)

            self.load_file_name = "{}_{}".format(self.policy_name,seed)

        # Save model
        output_model = "model_4"
        self.save_path = PACKAGE_PATH+\
            "config/{}/{}/".format(self.policy_name,output_model)

        self.save_file_name = "{}_{}".format(self.policy_name,self.seed)
        self.check_save_path = self.Check_Path(self.save_path)

        # load buffer
        if(self.load_buffer):
            load_buffer_model = "model_1"
            self.load_buffer_episode = 0
            self.load_buffer_path = PACKAGE_PATH+\
                "config/{}/{}/".format(self.policy_name,load_buffer_model)

    def Define_Action(self,r_info = 7,s_info = 360, a_info='angle'):
        if(a_info == 'angle'):
            self.state_dim = [r_info,s_info]
            self.action_dim = 1
            self.action_bound = [math.pi]
        elif(a_info == 'scalar_ang'):
            self.state_dim = [r_info,s_info]
            self.action_dim = 2
            self.action_bound = [30.0,math.pi]
        elif(a_info == 'linear'):
            self.state_dim = [r_info,s_info]
            self.action_dim = 2
            self.action_bound = [30.0,30.0]
        elif(a_info == 'linear_yaw'):
            self.state_dim = [r_info,s_info]
            self.action_dim = 3
            self.action_bound = [30.0,30.0,10.0]

    def Build_Env(self,robot,choose = 1,a_info = 'angle',dynamic = False):
        if(choose == 0):
            from lib.env.avoid_env_soccer import NubotAvoidEnv
            self.env = NubotAvoidEnv(robot_name = robot,\
                                     state_dim = self.state_dim,\
                                     action_dim = self.action_dim,\
                                     seed = self.seed)
        elif(choose == 1):
            from lib.env.avoid_env_soccer_1 import NubotAvoidEnv
            self.env = NubotAvoidEnv(robot_name = robot,\
                                     state_dim = self.state_dim,\
                                     action_dim = self.action_dim,\
                                     seed = self.seed,\
                                     save_route = self.save_route,\
                                     batch_file = self.batch_file)
        elif(choose == 2):
            from lib.env.avoid_env_soccer_2 import NubotAvoidEnv
            self.env = NubotAvoidEnv(robot_name = robot,\
                                     state_dim = self.state_dim,\
                                     action_dim = self.action_dim,\
                                     a_info = a_info,\
                                     dynamic = dynamic,\
                                     seed = self.seed,\
                                     save_route = self.save_route,\
                                     batch_file = self.batch_file)
        elif(choose == 3):
            from lib.env.car_env_soccer import Avoid_Soccor_Env
            self.env = Avoid_Soccor_Env(state_dim = self.state_dim,\
                                        action_dim = self.action_dim,\
                                        a_info = a_info,\
                                        reset = False,\
                                        seed = self.seed,\
                                        batch_file = self.batch_file)
    def Build_Policy(self):
        if(self.policy_name == 'sac_1'):
            from lib.network.sac_1.sac import SAC  
        if(self.policy_name == 'sac_5'):
            from lib.network.sac_5.sac import SAC
  
        self.policy = SAC(np.sum(self.state_dim),self.action_dim,self.action_bound,self.save_path)
        
        if(self.load_models):
            self.policy.Load(self.load_path,self.load_file_name)
    
    def Check_Path(self,path):
        if (not os.path.exists(path)):
            os.makedirs(path)
        return True
    
    def Play(self):
        total_timesteps = 0
        episode_tmp = 0
        timesteps_since_eval = 0

        avg_reward = 0
        episode_reward = 0

        good_log = 0
        bad_log = 0
        
        """ training procedure """
        for episode in range(1,self.max_episodes+1):
            state = self.env.Reset(self.reset_flag)
            print('run episode {} rate {} '.format(episode,good_log),end="")
            # print('run episode {} rate {} '.format(episode,good_log))
            """ 20hz """
            rate = rospy.Rate(20)
            for step in range(1,self.max_timesteps+1):
                if(self.render):
                    self.env.render()
    
                total_timesteps += 1
    
                action = self.policy.Select_Action(state,self.train)
                
                """ add noise """
                if(self.expl_noise != 0):
                    noise = np.random.normal(0,self.expl_noise,size=self.action_dim)
                    if(self.a_info == 'angle'):
                        action = (action+noise).clip(-self.action_bound[0],self.action_bound[0])
                    elif(self.a_info == 'scalar_ang'):
                        action[0] = (action[0]+noise[0]).clip(-self.action_bound[0],self.action_bound[0])
                        action[1] = (action[1]+noise[1]).clip(-self.action_bound[1],self.action_bound[1])
                    elif(self.a_info == 'linear'):
                        action = (action+noise).clip(-self.action_bound[0],self.action_bound[0])
                    elif(self.a_info == 'linear_yaw'):
                        action[0] = (action[0]+noise[0]).clip(-self.action_bound[0],self.action_bound[0])
                        action[1] = (action[1]+noise[1]).clip(-self.action_bound[1],self.action_bound[1])
                        action[2] = (action[2]+noise[2]).clip(-self.action_bound[2],self.action_bound[2])

                new_state,reward,done,info = self.env.Step(action)
                
                if(self.tutor):
                    self.nh.Pub_Tutor(state,action,reward,new_state,done)

                avg_reward += reward
                episode_reward += reward

                """ store """
                if(self.train):
                    self.replay_buffer.Add(state,\
                                        np.reshape(action,(self.action_dim,)),\
                                        reward,\
                                        new_state,\
                                        done)
                state = new_state
                timesteps_since_eval += 1

                if(self.train):
                    if(total_timesteps > self.start_timesteps):
                        # if(step % 2 == 0):
                        self.policy.Train(self.replay_buffer,total_timesteps-self.start_timesteps)

                if(done or step == (self.max_timesteps - 1)):
                    self.env.Stop_Env()
                    # if(self.train):
                    #     if(total_timesteps > self.start_timesteps):
                    #         loss = self.policy.Train(self.replay_buffer,step,episode-episode_tmp)
                    #     else:
                    #         episode_tmp += 1

                    self.policy.Log(episode_reward,episode)  
                    if(info == 'goal'):
                        good_log += 1
                        bad_log = 0
                        self.reset_flag = 3
                    # elif(bad_log >= 10):
                    #     bad_log = 0
                    #     self.reset_flag = 3
                    # else:
                    #     bad_log += 1
                    #     self.reset_flag = 0

                    # if(episode % 100 == 1):
                    #     self.reset_flag = 3
                    # else:
                    #     self.reset_flag = 6

                    break
                
                rate.sleep()
            
            """ save good job """
            if(self.train):
                self.Log(episode, episode_reward, step,info)
                if(episode % self.log_interval == 0 and total_timesteps > self.start_timesteps):
                    if(good_log >= self.good_job):
                    # avg_tmp = avg_reward / self.log_interval
                    # if(avg_tmp > -3):
                        print("#########################")
                        print("success episode ", episode)
                        print("save model")
                        print("#########################")
                        self.Save(episode)

                if(episode % self.eval_freq == 0):
                    self.Save(episode)
            else:
                self.Log_Show(episode, episode_reward, step, info)
            
            episode_reward = 0
            
            if(episode % self.log_interval == 0):
                avg_reward /= self.log_interval
                print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
                avg_reward = 0
                good_log = 0

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

def main():
    rospy.init_node('nubot_avoid_figure', anonymous=True)
    game = PlayGame()

    game.Play()


if __name__ == '__main__':
    main()