#!/usr/bin/env python3tf.global_variables('Q_value1')
# -*- coding: utf-8 -*-+

""" ros """
import rospy
import rospkg
from sensor_msgs.msg import LaserScan

""" lib """
from lib.gazebo.robot_gazebo_env import RobotGazeboEnv
from lib.nh.sim_nodehandle_1 import SimNodeHandle

""" tool """
import math
import numpy as np
import copy
import pickle
import os
import time

r"""
    bug
        Check_Model_Overlapping        
"""

class NubotAvoidEnv(RobotGazeboEnv):
    def __init__(self,robot_name,state_dim,action_dim,scan_info,a_info,dynamic,seed,save_route,batch_file):
        super(NubotAvoidEnv,self).__init__(
            robot_name = robot_name,
            reset_world_or_sim = "SIMULATION"
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scan_info = scan_info
        self.a_info = a_info
        self.dynamic = dynamic

        self.robot_dim = state_dim[0]
        self.scan_dim = state_dim[1]

        self.nh = SimNodeHandle(robot_name,state_dim[1],a_info = a_info)

        self.modelDict = {'name':'',\
                          'pos':[0.0,0.0,0.0],\
                          'ang':0.0,\
                          'range':[0.0,0.0],\
                          'vec':[0.0,0.0],\
                          'vec_':[0.0,0.0]}
        
        self.robot = copy.deepcopy(self.modelDict)
        self.robot['name'] = robot_name

        self.save_route = save_route
        
        """ setup random seed """
        np.random.seed(seed)

        """ init """
        self.Init_Env()

        if(batch_file >= 0):
            path = rospkg.RosPack().get_path('nubot_strategy')+\
                   '/avoid_1/lib/train_batch/num_{}/'.format(self.obstacle_num)
            self.train_data = self.Read_Train_Data(path+'{}_env.pickle'.format(batch_file))

        if(save_route):
            self.Log_Env()

        print("nubot avoid env init")
    
    """ Init env """
    def Init_Env(self):
        """ robot """
        self.robot['range'] = [0.23,0.23]
        self.robot['ang'] = 180.0
        self.robot['pos'] = [3.0,0.0,0.01]
        
        """ goal  """
        self.goal = copy.deepcopy(self.modelDict)
        self.goal['name'] = 'avoid_goal'
        self.goal['pos'] = [-2.8,0.0,0.01]
        self.goal['range'] = [0.2,0.75]

        """ obstacles """
        self.obstacle_num = 6
        
        if(self.obstacle_num == 0):
            obstacles = [[]]
        elif(self.obstacle_num == 1):
            obstacles = [[0.0,0.0]]
        elif(self.obstacle_num == 2):
            obstacles = [[0.0,0.0],[0.0,1.5]]
        elif(self.obstacle_num == 3):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5]]
        elif(self.obstacle_num == 4):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5],[0.0,2.0]]
        elif(self.obstacle_num == 5):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5],[0.0,2.0],[0.0,-2.0]]
        elif(self.obstacle_num == 6):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5],[0.0,2.0],[0.0,-2.0],[1.0,0.0]]
        elif(self.obstacle_num == 7):
            obstacles = [[0.0,0.0],\
                        [0.0,1.5],\
                        [0.0,-1.5],\
                        [0.0,2.0],\
                        [0.0,-2.0],\
                        [1.0,0.0],\
                        [1.0,1.5]]
        elif(self.obstacle_num == 8):
            obstacles = [[0.0,0.0],\
                        [0.0,1.5],\
                        [0.0,-1.5],\
                        [0.0,2.0],\
                        [0.0,-2.0],\
                        [1.0,0.0],\
                        [1.0,1.5],\
                        [1.0,-1.5]]
        
        self.obstacles = []
        for i in range(self.obstacle_num):
            obstacle = copy.deepcopy(self.modelDict)
            obstacle['name'] = 'obstacle_' + str(i+1)
            obstacle['pos'][0] = obstacles[i][0]
            obstacle['pos'][1] = obstacles[i][1]
            obstacle['pos'][2] = 0.01
            obstacle['range'] = [0.23,0.23]
            self.obstacles.append(obstacle)
        
        """ obstacle limit """
        self.ob_range_x = [-2.0,2.0]
        self.ob_range_y = [-1.77,1.77]
        self.ob_range_yaw = [-180,180]

        # test
        # self.ob_range_x = [-2.0,2.0]
        # self.ob_range_y = [-1.5,1.5]

        """ range limit """
        self.range_x = [-3.0,3.1]
        self.range_y = [-2.0,2.0]
        self.range_yaw = [-180,180]

        """ models """
        self.models = []

        """ goal count """
        self.goal_count = 0
        self.bad_count = 0
        self.pre_dis = 0
        self.init_pos = np.zeros(2)
        self.start_time = time.time()
        self.pre_front = 0
        self.step = 1

        if(self.scan_info == 'pre'):
            self.pre_scan = np.ones(self.scan_dim)*2.5
    
    def Read_Train_Data(self,path):
        with open(path,'rb') as f:
            data = pickle.load(f)
            return data

    def Log_Env(self):
        self.log_obstacle = []
        self.log_robot = []
        self.log_goal = []
        self.log_robot_path = []
        self.robot_path = []
    
    def Save_Log(self,directory):
        with open(directory+"log_obstacle_show.pickle",'wb') as f:
            pickle.dump(self.log_obstacle,f)
        with open(directory+"log_robot_show.pickle",'wb') as f:
            pickle.dump(self.log_robot,f)
        with open(directory+"log_path_show.pickle",'wb') as f:
            pickle.dump(self.log_robot_path,f)
        with open(directory+"log_goal_show.pickle",'wb') as f:
            pickle.dump(self.log_goal,f)
        print('save_log')

    """ check connect """
    def _Check_All_System_Ready(self):
        self.nh.Check_Connection()
        self.Check_Model_States()
    
    def _Init_Pose(self,reset = 1):
        r""" reset
            reset = 0 -> repeat pos
            reset = 1 -> random obstacle  - Challenge goal and robot
            reset = 2 -> randorm robot and obstacle - Challenge goal
            reset = 3 -> all random
            reset = 4 -> random goal  - current robot and current obstacle
            reset = 5 -> random robot  -  current obstacle and current goal
            reset = 6 -> random goal and robot - current obstacle
            reset = 7 -> read train batch - random goal and random robot
        """
        if(reset == 7):
            """ obstacle """
            box = self.Init_Obstacle_Pose([],'read')
            """ robot """
            robot = self.Init_Robot_Pose(box)
            """ goal """
            goal = self.Init_Goal_Pose(box)

            self.models = []
            self.models.append(robot)
            self.models.append(goal)
            for item in box:
                self.models.append(item)
        elif(reset == 6):
            """ obstacles """
            box = self.Init_Obstacle_Pose(self.models,'fixed')
            """ robot """
            robot = self.Init_Robot_Pose(box)
            """ goal """
            goal = self.Init_Goal_Pose(box)

            if(len(self.models)):
                self.models[0] = robot
                self.models[1] = goal
            else:
                self.models = []
                self.models.append(robot)
                self.models.append(goal)
                for item in box:
                    self.models.append(item)
        elif(reset == 5):
            """ obstacles """
            box = self.Init_Obstacle_Pose(self.models,'fixed')
            """ goal """
            goal = self.Init_Goal_Pose(box,'fixed')
            """ robot """
            robot = self.Init_Robot_Pose(box)

            if(len(self.models)):
                self.models[0] = robot
            else:
                self.models = []
                self.models.append(robot)
                self.models.append(goal)
                for item in box:
                    self.models.append(item)
        elif(reset == 4):
            """ obstacles """
            box = self.Init_Obstacle_Pose(self.models,'fixed')
            """ robot """
            robot = self.Init_Robot_Pose(box,'current')
            """ goal """
            goal = self.Init_Goal_Pose(box)

            if(len(self.models)):
                self.models[1] = goal
            else:
                self.models = []
                self.models.append(robot)
                self.models.append(goal)
                for item in box:
                    self.models.append(item)
        elif(reset == 3):
            # models = []
            # """ robot """
            # models.append(self.Init_Robot_Pose(models))
            # """ goal """
            # models.append(self.Init_Goal_Pose(models))
            # """ obstacle """
            # self.models = self.Init_Obstacle_Pose(models)
            
            models = []
            """ obstacle """
            box = self.Init_Obstacle_Pose(models)
            """ robot """
            robot = self.Init_Robot_Pose(box)
            """ goal """
            goal = self.Init_Goal_Pose(box)
            """ merge """
            self.models = []
            self.models.append(robot)
            self.models.append(goal)
            for item in box:
                self.models.append(item)
        elif(reset == 2):
            models = []
            """ robot """
            models.append(self.Init_Robot_Pose(models))
            """ goal """
            models.append(self.Init_Goal_Pose(models,'challenge'))
            """ obstacle """
            self.models = self.Init_Obstacle_Pose(models)
        elif(reset == 1):
            models = []
            """ robot """
            models.append(self.Init_Robot_Pose(models,'challenge'))
            """ goal """
            models.append(self.Init_Goal_Pose(models,'challenge'))
            """ obstacle """
            self.models = self.Init_Obstacle_Pose(models)
        elif(reset == 0):
            """ obstacle """
            box = self.Init_Obstacle_Pose(self.models,'fixed')
            """ goal """
            goal = self.Init_Goal_Pose(box,'fixed')
            """ robot """
            robot = self.Init_Robot_Pose(box,'fixed')

        if(self.save_route):
            self.log_robot.append(self.models[0])
            self.log_goal.append(self.models[1])
            self.log_obstacle.append(self.models[2:])

    def Init_Robot_Pose(self,models,state = 'random'):
        if(state == 'random'):
            while True:
                x,y,angle = self.Random_Pose('robot')
                if(self.Check_Model_Overlapping([x,y],models,error = 0.7) == False):
                    self.robot['pos'] = [x,y,0.01]
                    # self.robot['ang'] = angle
                    self.robot['ang'] = -180.0

                    self.init_pos[0] = x
                    self.init_pos[1] = y

                    self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])
                    return [x,y]
        elif(state == 'fixed'):
            if(len(self.models)):
                self.robot['ang'] = -180.0
                # _,_,angle = self.Random_Pose('robot')
                # self.robot['ang'] = angle

                self.robot['pos'][0] = self.init_pos[0]
                self.robot['pos'][1] = self.init_pos[1]

                self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])
                return [self.init_pos[0],self.init_pos[1]]
            else:
                return self.Init_Robot_Pose(models,'random')
        elif(state == 'current'):
            if(len(self.models)):
                self.init_pos[0] = self.robot['pos'][0]
                self.init_pos[1] = self.robot['pos'][1]

                self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])
                return [self.init_pos[0],self.init_pos[1]]
            else:
                return self.Init_Robot_Pose(models,'random')
        elif(state == 'challenge'):
            self.robot['ang'] = 180.0
            self.robot['pos'] = [3.0,0.0,0.01]
            self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])
            return [3.0,0.0]
        elif(state == 'read'):
            pass
    
    def Init_Goal_Pose(self,models,state='random'):
        if(state == 'random'):
            self.goal['range'] = [0.2,0.2]
            while True:
                x,y,_ = self.Random_Pose('robot')
                if(self.Check_Model_Overlapping([x,y],models,error = 0.7) == False):
                    self.goal['pos'] = [x,y,0.01]

                    print('random goal {}'.format([x,y]))
                    return [x,y]
        elif(state == 'fixed'):
            self.goal['range'] = [0.2,0.2]
            if(len(self.models)):
                self.goal['pos'] = [*self.models[1],0.01]
                
                print('fixed goal {}'.format(self.models[1]))
                return [self.models[1][0],self.models[1][1]]
            else:
                return self.Init_Goal_Pose(models,'random')
        elif(state == 'challenge'):
            self.goal['pos'] = [-2.8,0.0,0.01]
            self.goal['range'] = [0.2,0.75]
        
            print('challenge goal')
            return [-2.8,0.0]
        elif(state == 'read'):
            pass
    
    def Init_Obstacle_Pose(self,models,state='random'):
        if(state == 'random'):
            box = []
            for item in models:
                box.append(item)
            i = 0
            while i < self.obstacle_num:
                x,y,angle = self.Random_Pose('obstacle')
                if(self.Check_Model_Overlapping([x,y],box) == False):
                    self.obstacles[i]['pos'] = [x,y,0.01]
                    # self.obstacles[i]['ang'] = 0
                    self.obstacles[i]['ang'] = angle
                    
                    box.append([x,y])

                    self.Set_Model(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    i += 1
            return box
        elif(state == 'fixed'):
            if(len(self.models)):
                box = []
                for i in range(len(self.models[2:])):
                    self.obstacles[i]['pos'] = [*self.models[2+i],0.01]
                    self.obstacles[i]['ang'] = 0

                    # _,_,angle = self.Random_Pose('obstacle')
                    # self.obstacles[i]['ang'] = angle

                    box.append(self.obstacles[i]['pos'])
                    self.Set_Model(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                return box
            else:
                return self.Init_Obstacle_Pose(models,'random')
        elif(state == 'read'):
            box = []
            idx = np.random.randint(7777)%len(self.train_data)
            data = self.train_data[idx]

            for i in range(self.obstacle_num):
                # _,_,angle = self.Random_Pose('obstacle')
                # self.obstacles[i]['ang'] = angle

                self.obstacles[i]['pos'] = [*data[i],0.01]
                box.append(data[i])
                self.Set_Model(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])

            return box
                    

    def _Init_Robot_Control(self):
        self.nh.Pub_Cmdvel([0,0,0])

    def _Init_Control(self):
        # self._Check_All_System_Ready()
        self.Check_Robot_Stop()

    def _Init_Dynamic_Env(self):
        if(self.dynamic):
            self.nh.Reset_Dynamic_Env()
    """ process """
    def Step(self,action):

        if(self.a_info == 'angle'):
            """ scalar ang yaw = 0"""
            scalar = np.array([30.0])
            action_ = np.concatenate((scalar,action),axis=None)
            action_ = np.concatenate((action_,0),axis=None)
        elif(self.a_info == 'scalar_ang'):
            """ scalar ang yaw = 0 """
            action_ = np.concatenate((action,0),axis=None)
        elif(self.a_info == 'linear'):
            """ x y yaw = 0 """
            action_ = np.concatenate((action,0),axis=None)
        elif(self.a_info == 'linear_yaw'):
            """ x y yaw """
            action_ = copy.deepcopy(action)
            
        self.Start_Sim()
        self.nh.Pub_Cmdvel(action_,len(action))
        # print(time.time())
        self.nh.Sub_Scan()
        
        obs = self.Get_Obs()
        done,info = self.Is_Done(obs)
        reward,info = self.Compute_Reward(obs,action_,info)

        if(self.save_route):
            self.robot_path.append(obs[:2])
        if(done and self.save_route):
            self.log_robot_path.append(self.robot_path)

        return obs,reward,done,info
    
    def Get_Obs(self,reset = False):
        if(reset):
            self.Start_Sim()
            self.nh.Sub_Scan()
            self.Sub_Robot_Model()
            self.Stop_Sim()

        self.Update_Robot_By_Model()
        
        """ pos """
        # robot = np.concatenate((np.array(self.robot['pos'][:2]),np.array([math.radians(self.robot['ang'])])))
        # pos = np.concatenate((robot,np.array(self.goal['pos'][:2])))
        # obs = np.hstack((pos,self.nh.scan))

        
        """ dis angle """
        r_pos = self.robot['pos'][:2]
        r_ang = np.array([math.radians(self.robot['ang'])])
        p_robot = np.concatenate((r_pos,r_ang),axis=0)

        dis = self.Cal_Goal_Robot_Dis()

        ang = self.Cal_Goal_Robot_Ang('rad')

        robot_state = np.concatenate((p_robot,np.array([dis,ang])),axis=0)

        # add robot vec
        # robot_state = np.concatenate((robot_state,self.robot['vec']),axis=0)

        # add robot vec_ vec 
        robot_state = np.concatenate((robot_state,self.robot['vec_']),axis=0)
        robot_state = np.concatenate((robot_state,self.robot['vec']),axis=0)

        if(self.scan_info == 'current'):
            obs = np.hstack((robot_state,self.nh.scan))
        elif(self.scan_info == 'pre'):
            scan = copy.deepcopy(self.nh.scan)
            diff_scan = scan - self.pre_scan
            obs = np.hstack((robot_state,scan))
            obs = np.hstack((obs,diff_scan))
            self.pre_scan = scan
        
        # print(obs[:self.robot_dim])
        return obs
    
    def Is_Done(self,obs):

        info = 'None'
        if(self.Check_Arrive_GoalAera()):
            # if(self.goal_count >= 50):
            #     return True,'goal'
            # else:
            #     self.goal_count += 1
            #     return False,'goal'
            return True,'goal'

        if(self.nh.bumper):
            self.nh.Reset_Bumper()
            # if(self.bad_count >= 50):
            #     return True,'bump'
            # else:
            #     self.bad_count += 1
            #     return False,'bump'
            return True,'bump'
            # return False,'bump'

        if(self.Check_Robot_Over_Range(self.robot['pos'][:2])):
            # if(self.bad_count >= 50):
            #     return True,'over range'
            # else:
            #     self.bad_count += 1
            #     return False,'over range'
            return True,'over range'
            # return False,'over range'

        
        return False,info
    
    def Compute_Reward(self,obs,action,info):
        
        if(self.a_info == 'angle'):
            angle = action[1]
        elif(self.a_info == 'scalar_ang'):
            angle = action[1]
        elif(self.a_info == 'linear'):
            angle = math.atan2(action[1],action[0])
            v = math.hypot(*action[:2])
        elif(self.a_info == 'linear_yaw'):
            angle = math.atan2(action[1],action[0])
            v = math.hypot(*action[:2])
            theta = action[2]

        dis = self.Cal_Goal_Robot_Dis()

        """ reward for angle"""
        """ reward 1 """

        # reward = -dis/40.0
        # reward += round(4.76*(1.05-abs(obs[4]-angle)),2)
        # reward -= 0.01*self.step
        # self.step += 1

        # if(info == 'goal' and self.goal_count >= 50):
        #     reward = 20.
        # elif(info == 'goal' and self.goal_count < 50):
        #     reward = 2.
        # elif(info == 'bump'):
        #     reward = -20.
        # elif(info == 'over range'):
        #     reward = -20.

        # return reward,info

        """ reward 2 """

        # reward = (-dis+1)/10.0
        # reward += 0.382*(0.262-abs(obs[4]-angle))
        # reward -= 0.01*self.step
        # self.step += 1

        # if(info == 'goal'):
        #     reward = 3.
        # elif(info == 'goal' and self.goal_count < 50):
        #     reward = 2.
        # elif(info == 'bump'):
        #     reward = -1.
        # elif(info == 'over range'):
        #     reward = -1.
        # return reward,info

        """ reward 3 for APF"""

        # c_att = 4.5
        # c_rep = 0.1
        # error = 0.8

        # if(dis <= error):
        #     u_att = 0.5*c_att*dis*dis
        # else:
        #     u_att = error*c_att*dis - 0.5*c_att*error*error
        
        # u_rep = c_rep*self.Cal_Scan_Reward(obs,angle,error=1.0)

        # reward = (3-u_att) + u_rep
        # reward += round(4.76*(1.05-abs(obs[4]-angle)),2)
        # reward -= 0.01*self.step
        # self.step += 1
        
        # print((8-u_att),u_rep)
        # if(info == 'goal' and self.goal_count >= 50):
        #     reward = 200.
        # elif(info == 'goal' and self.goal_count < 50):
        #     reward += 1.
        # elif(info == 'bump'):
        #     reward = -100.
        # elif(info == 'over range'):
        #     reward = -100.
        # return reward,info

        """ reward 4 """
        # dis_o = min(obs[7:])

        # reward = (-dis+1)/20.0
        # reward += 0.1274*(1.57-abs(obs[4]-angle))
        # if(dis_o < 0.3):
        #     reward -= 0.5
        # elif(dis_o < 0.4):
        #     reward -= 0.3
        
        # if(info == 'goal'):
        #     reward = 2.
        # elif(info == 'bump'):
        #     reward = -1.
        # elif(info == 'over range'):
        #     reward = -1.
        # return reward,info

        """ reward 5 """
        # dis_o = min(obs[7:])

        # reward = 0
        # reward -= dis
        # reward += 0.4
        
        # if(reward > 0):
        #     reward *= 2

        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward += (cos_vec*dis - dis)
        # reward -= 1.0

        # if(info == 'goal'):
        #     reward = 10.
        # elif(info == 'bump'):
        #     reward = -10.
        # elif(info == 'over range'):
        #     reward = -10.
        # elif(dis_o < 0.3):
        #     reward -= 8.
        # elif(dis_o < 0.4):
        #     reward -= 5.

        # return reward,info
        """ reward 6 """
        # dis_o = min(obs[self.robot_dim:])
        # x1 = abs(self.range_x[0]-self.robot['pos'][0])
        # x2 = abs(self.range_x[1]-self.robot['pos'][0])
        # y1 = abs(self.range_y[0]-self.robot['pos'][1])
        # y2 = abs(self.range_y[1]-self.robot['pos'][1])
        # dis_f = round(min([x1,x2,y1,y2]),3)
        # bad_dis = min([dis_o,dis_f])

        # reward = (-dis+0.5)/10
        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward += (cos_vec*dis - dis)/10
        
        # if(reward > 0):
        #     reward *= 3

        # if(bad_dis < 0.3):
        #     reward -= 3
        # elif(bad_dis < 0.4):
        #     reward -= 2
        # elif(bad_dis < 0.5):
        #     reward -= 1
        # elif(bad_dis < 0.6):
        #     reward -= 0.5
        # elif(bad_dis < 0.8):
        #     reward -= 0.3

        # if(info == 'goal'):
        #     reward = 5.
        # elif(info == 'bump'):
        #     reward = -5.
        # elif(info == 'over range'):
        #     reward = -10.
        # return reward,info

        """ reward better """
        # reward = -dis+0.3
        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward += (cos_vec*0.2-0.01)
        
        # if(info == 'goal'):
        #     reward = 100.
        # elif(info == 'bump'):
        #     reward = -100.
        # elif(info == 'over range'):
        #     reward = -100.

        # return reward,info
        
        """ reward for linear"""
        """ reward 1 """
        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward = 0.2*(v-15)/15 + (v/30)*cos_vec*0.2 - 0.05
        # # reward -= 0.001*self.step
        # # self.step += 1
        # if(info == 'goal'):
        #     reward = 100.
        # elif(info == 'bump'):
        #     reward = -1.
        # elif(info == 'over range'):
        #     reward = -1.
        # return reward,info

        """ reward 2 """
        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward = 0.2*(v-15)/15 + (v/30)*cos_vec*0.2 - 0.05
        # # reward -= 0.001*self.step
        # # self.step += 1
        # if(info == 'goal'):
        #     reward = 100.
        # elif(info == 'bump'):
        #     reward += 0.
        # elif(info == 'over range'):
        #     reward += 0.
        # return reward,info

        """ reward 3 """
        # goal
        # rg = -0.2 + 10*(self.pre_dis-dis)
        # # avoid
        # # phi = abs(obs[4]-angle)
        # phi = abs(angle-self.pre_front)
        # if(phi > math.pi):
        #     ro_a = 0.3*((v-15)/30)
        # else:
        #     ro_a = 0.3*((v-15)/30)+np.cos(phi)*0.2

        # dis_o = np.min(obs[self.robot_dim:])
        # if(0.2 < dis_o <= 0.5):
        #     ro_s = -0.1/(dis_o**2+1e-6)
        # elif(dis_o > 0.5):
        #     ro_s = 0
        
        # reward = rg + ro_a + ro_s
        
        # if(info == 'goal'):
        #     reward = 10.
        # elif(info == 'bump'):
        #     reward = -10.
        # elif(info == 'over range'):
        #     reward = -10.

        # self.pre_dis = dis
        # self.pre_front = angle

        # return reward,info
        """ reward 4 """
        # goal
        # rg = -0.2 + 10*(self.pre_dis-dis)

        # # avoid
        # # phi = abs(obs[4]-angle)
        # phi = abs(angle-self.pre_front)
        # if(phi > math.pi):
        #     ro_a = 0.2*((v-25)/25)
        # else:
        #     ro_a = 0.2*((v-25)/25)+np.cos(phi)*0.3

        # if(self.scan_info == 'current'):
        #     dis_o = np.min(obs[self.robot_dim:])
        # elif(self.scan_info == 'pre'):
        #     dis_o = np.min(obs[self.robot_dim:-self.scan_dim])
        # if(0.1 < dis_o <= 0.5):
        #     ro_s = -0.1/((dis_o - 0.13)**2+1e-6)
        # else:
        #     ro_s = 0
        
        # # merge
        # fp = min(0.1/((dis_o - 0.13)**2 +1e-6),4)
        # fp_inv = max(min(1/(fp+1e-6),2),0.1)
        # reward = fp_inv*rg + fp*(ro_a + ro_s)

        # if(info == 'goal'):
        #     reward = 10.
        # elif(info == 'bump'):
        #     reward = -10.
        # elif(info == 'over range'):
        #     reward = -10.

        # self.pre_dis = dis
        # self.pre_front = angle

        # return reward,info

        """ reward 5 """
        # goal
        rg = -0.2 + 10*(self.pre_dis-dis)

        # avoid
        # phi = abs(obs[4]-angle)
        phi = abs(angle-self.pre_front)

        v = min(v,70.0)
        if(phi > math.pi):
            ro_a = -0.25+0.15*((v-25)/35)
        else:
            ro_a = -0.25+0.15*((v-25)/35)+np.cos(phi)*0.3

        if(self.scan_info == 'current'):
            dis_o = np.min(obs[self.robot_dim:])
        elif(self.scan_info == 'pre'):
            dis_o = np.min(obs[self.robot_dim:-self.scan_dim])
        # if(dis_o <= 0.5):
        #     ro_s = -0.1/((dis_o - 0.13)**2+1e-6)
        # else:
        #     ro_s = 0

        ro_s = 0
        
        # merge
        fp = min(0.1/((dis_o - 0.13)**2 +1e-6),4)
        fp_inv = max(min(1/(fp+1e-6),2),0.1)
        reward = fp_inv*rg + fp*(ro_a + ro_s)

        if(info == 'goal'):
            reward = 10.
        elif(info == 'bump'):
            reward = -10.
        elif(info == 'over range'):
            reward = -15.

        self.pre_dis = dis
        self.pre_front = angle

        return reward,info

        """ reward for linear yaw"""
        """ reward 1 """
        # cos_vec = np.cos(math.radians(theta*18))
        # reward = 0.2*(v-15)/15 + (v/30)*cos_vec*0.2 - 0.05
        # if(info == 'goal'):
        #     reward = 100.
        # elif(info == 'bump'):
        #     reward = -10.
        # elif(info == 'over range'):
        #     reward = -10.
        # return reward,info
    
    def Stop_Env(self):
        self.Stop_Sim()

    def Reset(self,reset = 1):
        self.Reset_Sim(reset)
        self.nh.Reset_Bumper()
        
        self.goal_count = 0
        self.bad_count = 0
        self.pre_dis = self.Cal_Goal_Robot_Dis()
        self.pre_front = self.robot['ang']
        self.start_time = time.time()
        self.step = 1
        """ log """
        if(self.save_route):
            self.robot_path = []

        return self.Get_Obs(True)

    """ tool """
    def Random_Pose(self,model,ob_error = 0.01,r_error = 0.25):
        if(model == 'obstacle'):
            x = np.random.uniform(self.ob_range_x[0]+ob_error,self.ob_range_x[1]-ob_error,1)
            y = np.random.uniform(self.ob_range_y[0]+ob_error,self.ob_range_y[1]-ob_error,1)
            angle = np.random.uniform(*self.ob_range_yaw,1)
        elif(model == 'robot'):
            x = np.random.uniform(self.range_x[0]+r_error,self.range_x[1]-r_error,1)
            y = np.random.uniform(self.range_y[0]+r_error,self.range_y[1]-r_error,1)
            angle = np.random.uniform(*self.range_yaw,1)
        return round(x[0],3),round(y[0],3),round(angle[0])

    """ bug """
    # def Check_Model_Overlapping(self,pos,models,error = 0.70):
    # def Check_Model_Overlapping(self,pos,models,error = 0.8):
    # def Check_Model_Overlapping(self,pos,models,error = 1.2):
    def Check_Model_Overlapping(self,pos,models,error = 1.27):
        """ method 2 """
        for item in models:
            if(math.hypot(pos[0]-item[0],pos[1]-item[1]) < error):
                return True
        return False
    
    def Check_Robot_Over_Range(self,pose):
        if(pose[0] <= self.range_x[0]):
            return True
        if(pose[0] >= self.range_x[1]):
            return True
        if(pose[1] <= self.range_y[0]):
            return True
        if(pose[1] >= self.range_y[1]):
            return True
        return False

    def Check_Arrive_GoalAera(self):
        
        range_x1 = (self.goal['pos'][0]-self.goal['range'][0])
        range_x2 = (self.goal['pos'][0]+self.goal['range'][0])
        range_y1 = (self.goal['pos'][1]-self.goal['range'][1])
        range_y2 = (self.goal['pos'][1]+self.goal['range'][1])

        if(self.robot['pos'][0] < range_x1):
            return False
        if(self.robot['pos'][0] > range_x2):
            return False
        if(self.robot['pos'][1] < range_y1):
            return False
        if(self.robot['pos'][1] > range_y2):
            return False
        return True

    def Update_Robot_By_Model(self):
        self.robot['pos'][0] = self.robotPos[0]
        self.robot['pos'][1] = self.robotPos[1]
        self.robot['ang'] = self.robotPos[2]

        self.robot['vec_'][0] = self.robot['vec'][0]
        self.robot['vec_'][1] = self.robot['vec'][1]

        self.robot['vec'][0] = self.robotVec[0]
        self.robot['vec'][1] = self.robotVec[1]

    def Cal_Goal_Robot_Dis(self):
        return np.hypot(*(np.array(self.robot['pos'][:2]) - np.array(self.goal['pos'][:2])))
    
    def Cal_Goal_Robot_Ang(self,ang_rad = 'deg'):
        dx = self.goal['pos'][0] - self.robot['pos'][0]
        dy = self.goal['pos'][1] - self.robot['pos'][1]
        ang = math.degrees(math.atan2(dy,dx))
        ang -= self.robot['ang']
        ang = self.Norm_Angle(ang)

        if(ang_rad == 'deg'):
            return ang
        elif(ang_rad == 'rad'):
            return math.radians(ang)

    def Norm_Angle(self,angle):
        if(angle > 180):
            angle -=360
        elif(angle < -180):
            angle +=360
        return angle
    
    def Cal_Scan_Reward(self,obs,angle,error = 1.6,ang_range = 60):
        """ method 6 for APF """
        """ obstacle """
        r_error = self.robot['range'][0]-0.02
        dis_o = min(obs[self.robot_dim:])

        """ field """
        x1 = abs(self.range_x[0]-self.robot['pos'][0])
        x2 = abs(self.range_x[1]-self.robot['pos'][0])
        y1 = abs(self.range_y[0]-self.robot['pos'][1])
        y2 = abs(self.range_y[1]-self.robot['pos'][1])
        dis_f = round(min([x1,x2,y1,y2]),3)
        
        dis = min([dis_o,dis_f])-r_error
        if(dis <= 0.08):
            dis = 0.08

        if(dis <= error):
            reward = 0.5*math.pow((1/dis)-(1/error),2.0)
        else:
            reward = 0
        return -reward
    
    def Store_Buffer(self,buffer):
        self.nh.Store_Buffer(buffer)
    

        
        
                

        