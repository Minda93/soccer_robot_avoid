#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" ros """
import rospy
from sensor_msgs.msg import LaserScan

""" lib """
from lib.gazebo.robot_gazebo_env import RobotGazeboEnv
from lib.nh.sim_nodehandle import SimNodeHandle

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
    def __init__(self,robot_name,state_dim,action_dim,seed):
        super(NubotAvoidEnv,self).__init__(
            robot_name = robot_name,
            reset_world_or_sim = "SIMULATION"
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.nh = SimNodeHandle(robot_name,state_dim[1])

        self.modelDict = {'name':'',\
                          'pos':[0.0,0.0,0.0],\
                          'ang':0.0,\
                          'range':[0.0,0.0],\
                          'vec':[0.0,0.0]}
        
        self.robot = copy.deepcopy(self.modelDict)
        self.robot['name'] = robot_name
        
        """ setup random seed """
        np.random.seed(seed)

        """ init """
        self.Init_Env()
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
        obstacle_num = 5
        
        if(obstacle_num == 1):
            obstacles = [[0.0,0.0]]
        elif(obstacle_num == 2):
            obstacles = [[0.0,0.0],[0.0,1.5]]
        elif(obstacle_num == 3):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5]]
        elif(obstacle_num == 4):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5],[0.0,2.0]]
        elif(obstacle_num == 5):
            obstacles = [[0.0,0.0],[0.0,1.5],[0.0,-1.5],[0.0,2.0],[0.0,-2.0]]
        elif(obstacle_num == 8):
            obstacles = [[0.0,0.0],\
                        [0.0,1.5],\
                        [0.0,-1.5],\
                        [0.0,2.0],\
                        [0.0,-2.0],\
                        [1.0,0.0],\
                        [1.0,1.5],\
                        [1.0,-1.5]]
        
        self.obstacles = []
        for i in range(obstacle_num):
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

        """ range limit """
        self.range_x = [-3.0,3.1]
        self.range_y = [-2.0,2.0]
        self.range_yaw = [-180,180]

        """ goal count """
        self.goal_count = 0
    
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

    """ check connect """
    def _Check_All_System_Ready(self):
        self.nh.Check_Connection()
        self.Check_Model_States()
    
    def _Init_Pose(self,reset = 1):
        if(reset == 3):
            models = []
            """ robot """
            # random
            x,y,angle = self.Random_Pose('robot')
            self.robot['pos'] = [x,y,0.01]
            # self.robot['ang'] = angle
            self.robot['ang'] = -180.0
            models.append([x,y])

            self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])

            """ goal """
            x,y,angle = self.Random_Pose('robot')
            self.goal['pos'] = [x,y,0.01]
            self.goal['range'] = [0.2,0.2]
            # self.goal['range'] = [0.5,0.5]
            models.append([x,y])
            print('random goal {}'.format([x,y]))
            """ obstacle """
            i = 0
            while i < len(self.obstacles):
                x,y,angle = self.Random_Pose('obstacle')
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    self.obstacles[i]['pos'] = [x,y,0.01]
                    # self.obstacles[i]['ang'] = angle
                    
                    models.append([x,y])

                    self.Set_Model(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    # print(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    i += 1
            self.log_robot.append(models[0])
            self.log_goal.append(models[1])
            self.log_obstacle.append(models[2:])

        elif(reset == 2):
            models = []
            """ robot """
            # random
            x,y,angle = self.Random_Pose('robot')
            self.robot['pos'] = [x,y,0.01]
            self.robot['ang'] = -180.0
            # self.robot['ang'] = angle
            models.append([x,y])

            self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])
            # print(self.robot['name'],self.robot['pos'],self.robot['ang'])

            """ goal """
            self.goal['pos'] = [-2.8,0.0,0.01]
            self.goal['range'] = [0.2,0.75]
            models.append([-2.8,0.0])

            """ obstacle """
            i = 0
            while i < len(self.obstacles):
                x,y,angle = self.Random_Pose('obstacle')
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    self.obstacles[i]['pos'] = [x,y,0.01]
                    # self.obstacles[i]['ang'] = angle
                    
                    models.append([x,y])

                    self.Set_Model(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    # print(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    i += 1
            self.log_robot.append(models[0])
            self.log_goal.append(models[1])
            self.log_obstacle.append(models[2:])
        elif(reset == 1):
            models = []
            # init
            self.robot['ang'] = 180.0
            self.robot['pos'] = [3.0,0.0,0.01]
            models.append([3.0,0.0])

            self.Set_Model(self.robot['name'],self.robot['pos'],self.robot['ang'])

            """ goal """
            self.goal['pos'] = [-2.8,0.0,0.01]
            self.goal['range'] = [0.2,0.75]
            models.append([-2.8,0.0])

            """ obstacle """
            i = 0
            while i < len(self.obstacles):
                x,y,angle = self.Random_Pose('obstacle')
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    self.obstacles[i]['pos'] = [x,y,0.01]
                    # self.obstacles[i]['ang'] = angle
                    
                    models.append([x,y])

                    self.Set_Model(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    # print(self.obstacles[i]['name'],self.obstacles[i]['pos'],self.obstacles[i]['ang'])
                    i += 1
            self.log_robot.append(models[0])
            self.log_goal.append(models[1])
            self.log_obstacle.append(models[2:])
        else:
            pass
    
    def _Init_Robot_Control(self):
        self.nh.Pub_Cmdvel([0,0,0])

    def _Init_Control(self):
        # self._Check_All_System_Ready()
        self.Check_Robot_Stop()

    """ process """
    def Step(self,action):
        if(type(action).__module__ != np.__name__):
            action = np.array(action)
        
        if(len(action) == 1):
            vector = np.array([30.0])
            action_ = np.concatenate((vector,action),axis=None)
            action_ = np.concatenate((action_,0),axis=None)
        elif(len(action) == 2):
            """ x y linear """
            # action_ = np.concatenate((action,0),axis=None)

            """ scalar ang """
            x,y = self.nh.Calculate_Vel(action)

            action_ = np.array([x,y,0])
        else:
            action_ = action
            
        self.Start_Sim()
        self.nh.Pub_Cmdvel(action_,len(action))
        # print(time.time())
        self.nh.Sub_Scan()
        # msg = None
        # while msg is None and not rospy.is_shutdown():
        #     try:
        #         msg = rospy.wait_for_message("/scan", LaserScan, timeout=5.0)
        #     except:
        #         print("don't listen scan")

        # self.Stop_Sim()

        obs = self.Get_Obs()
        done,info = self.Is_Done(obs)
        reward,info = self.Compute_Reward(obs,action_,info)

        self.robot_path.append(obs[:2])

        if(done):
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
        robot_state = np.concatenate((robot_state,self.robot['vec']),axis=0)
        
        obs = np.hstack((robot_state,self.nh.scan))

        # print(obs[:7])
        return obs
    
    def Is_Done(self,obs):

        info = 'None'
        if(self.nh.bumper):
            self.nh.Reset_Bumper()
            return True,'bump'
        if(self.Check_Robot_Over_Range(self.robot['pos'][:2])):
            return True,'over range'
        if(self.Check_Arrive_GoalAera()):
            if(self.goal_count >= 50):
                return True,'goal'
            else:
                self.goal_count += 1
                return False,'goal'
        
        return False,info
    
    def Compute_Reward(self,obs,action,info):
        dis = self.Cal_Goal_Robot_Dis()
        
        """ reward 1 """
        # reward = -dis/10.
        # 60 angle = 1.05 rad
        # reward range [-10,5]
        # if(self.action_dim == 1):
            # reward += round(4.85*(1.05-abs(action[1])),2)
            # reward += round(3.185*(1.57-abs(action[1])),2)
        
        # 60 angle = 1.05 rad
        # reward range [-10,5]
        # if(self.action_dim == 1):
        #     # reward += round(4.85*(1.05-abs(action[1])),2)
        #     reward += round(3.185*(1.57-abs(action[1])),2)
        # else:
        #     move_ang = math.atan2(action[1],action[0])
        #     reward += round(4.85*(1.05-abs(move_ang)),2)/5.
        #     # move
        #     reward += round(((np.hypot(action[0],action[1])-15)/10.0),2)
        # print("reward",reward)

        """ reward 3 120 is better"""
        reward = -dis/40.0
        # 120
        reward += round(4.76*(1.05-abs(obs[4]-action[1])),2)

        # 270
        # reward += round(2.12*(2.36-abs(obs[4]-action[1])),2)

        """ reward 4 """
        # reward = (-dis - 1.0)*2.0
        # reward += round(4.76*(1.05-abs(obs[4]-action[1])),2)

        """ reward 5 """
        # reward = - dis - 1.0

        """ reward 6 """
        # reward = (-dis - 1.0)
        # reward += round(4.76*(1.05-abs(obs[4]-action[1])),2)
        # reward += 3*(self.Cal_Scan_Reward(obs,action)/len(obs[7:]))
        
        if(info == 'goal' and self.goal_count >= 50):
            reward += 100.
        elif(info == 'goal' and self.goal_count < 50):
            reward += 10.
        elif(info == 'bump'):
            reward -= 500.
        elif(info == 'over range'):
            reward -= 500.
        
        return reward,info
    
    def Stop_Env(self):
        self.Stop_Sim()

    def Reset(self,reset = 1):
        self.Reset_Sim(reset)
        self.nh.Reset_Bumper()

        self.goal_count = 0

        """ log """
        self.robot_path = []

        return self.Get_Obs(True)

    """ tool """
    def Random_Pose(self,model,ob_error = 0.01,r_error = 0.15):
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
    def Check_Model_Overlapping(self,pos,models,error = 0.48):
        """ method 1 """
        # for item in models:
        #     flag = True
        #     if(pos[0] < (item[0] - error)):
        #         flag = False
        #     if(pos[0] > (item[0] + error)):
        #         flag = False
        #     if(pos[1] < (item[1] - error)):
        #         flag = False
        #     if(pos[1] < (item[1] + error)):
        #         flag = False
        #     if(flag):
        #         break
        # return flag
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
    
    def Cal_Scan_Reward(self,obs,action,error = 0.8):
        
        scan = np.zeros(len(obs[7:]))
        scan_ = np.zeros(len(scan))
        for i in range(len(obs[7:])):
            if(obs[i+7] >= error):
                scan[i] = 1
            else:
                scan[i] = -1
        
        length = math.ceil(len(scan)/2)-1
        num = 0
        for i in range(len(scan)):
            for j in range(1,length+1):
                front = (i+j)%len(scan)
                back = (i-j)%len(scan)
                if(scan[front] == scan[i] and scan[back] == scan[i]):
                    if(scan[i] > 0):
                        num += 1
                    else:
                        num -= 10
                else:
                    break
            scan_[i] = scan[i]+num
            num = 0
        
        ang = math.degrees(action[1])
        if(ang < 0):
            ang += 360
        idx = int(ang/(360/len(scan)))
        
        return scan_[idx]
        
                

        