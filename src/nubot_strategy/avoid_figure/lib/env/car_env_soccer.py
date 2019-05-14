# -*- coding: utf-8 -*-

""" sys """
import sys
# sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
# import cv2

""" sim """
import pyglet

""" ros """
import rospkg

""" tool """
import copy
import numpy as np
import time
import math
from numba import jit 
import pickle

""" lib """
from lib.env.pos_transfer import *

r""" bug
    Check_Model_Overlapping
    front scan
"""


class MAZE(object):
    def __init__(self,width = 700,height = 500,ob_num = 5):
        self.__width = width
        self.__height = height

        self.__ob_num = ob_num

        self.modelDict = {'name':'',\
                          'pos':[0.0,0.0],\
                          'ang':0.0,\
                          'range':[0.0,0.0]}

        """ goal """
        self.Init_Goal()
        self.Init_Obstacle()

    def Init_Goal(self):
        """ goal """
        self.__goal = copy.deepcopy(self.modelDict)
        self.__goal['name'] = 'goal'
        self.__goal['pos'] = np.array([-280,0])
        self.__goal['ang'] = 0.0
        self.__goal['range'] = np.array([20,70])

        self.__goal['o_pos'] = np.array([-280,0])

    def Init_Obstacle(self):
        # """ obstacle """
        if(self.__ob_num == 4):
            obstacles = [[0,0],[0,150],[0,-150],[0,200]]
        elif(self.__ob_num == 5):
            obstacles = [[0.0,0.0],\
                        [0.0,150],\
                        [0.0,150],\
                        [0.0,200],\
                        [0.0,-200]]
        elif(self.__ob_num == 8):
            obstacles = [[0.0,0.0],\
                        [0.0,150],\
                        [0.0,150],\
                        [0.0,200],\
                        [0.0,-200],\
                        [100,0.0],\
                        [100,105],\
                        [100,-105]]

        self.__obstacles = []
        for i in range(len(obstacles)):
            obstacle = copy.deepcopy(self.modelDict)
            obstacle['name'] = 'obstacle_' + str(i+1)
            obstacle['pos'] = obstacles[i]
            obstacle['range'] = np.array([23,23])
            self.__obstacles.append(obstacle)
    
    def Set_Goal_Pos(self,pos,ang = 0,range = None):
        self.__goal['pos'] = pos
        self.__goal['ang'] = ang
        if(range is None):
            self.__goal['range'] = np.array([20,70])
        else:
            self.__goal['range'] = range
    
    def Set_Obstacle_Pos(self,idx,pos,ang = 0):
        self.__obstacles[idx-1]['pos'] = pos
        self.__obstacles[idx-1]['ang'] = ang

    @property
    def width(self):
        return self.__width
    
    @property
    def height(self):
        return self.__height

    @property
    def ob_num(self):
        return self.__ob_num

    @property
    def goal(self):
        return self.__goal
    
    @property
    def obstacles(self):
        return self.__obstacles
    
    
class Car(object):
    def __init__(self,pos = [280,0],radius = 28,ang = -180,vec = 2,scan_num = 360,scan_dis = 250):
        self.modelDict = {'name':'',\
                          'pos':[0,0],\
                          'o_pos':[0,0],\
                          'radius':0,\
                          'ang':0.0,\
                          'o_ang':0.0,\
                          'vec':0,\
                          'scan_num':360,\
                          'scan_dis':0}
        self.__car = copy.deepcopy(self.modelDict)

        self.__car['name'] = 'robot'
        self.__car['pos'] = np.array(pos)
        self.__car['o_pos'] = np.array(pos)
        self.__car['init_pos'] = np.array(pos)
        self.__car['radius'] = radius
        self.__car["ang"] = ang
        self.__car["o_ang"] = ang
        self.__car['vec'] = vec
        self.__car['scan_num'] = scan_num
        self.__car['scan_dis'] = scan_dis

        self.__trajectory = None
    
    def Set_Car_Pos(self,pos,ang = 0):
        self.__car['pos'] = pos
        self.__car['ang'] = ang
    
    def Set_Car_Init_Pos(self,pos):
        self.__car['init_pos'] = pos
    
    def Set_Car_Trajectory(self,pos = [0,0],reset = False):
        if(reset):
            self.__trajectory = None
        else:
            if(self.__trajectory is None):
                self.__trajectory = np.array(pos)
            else:
                self.__trajectory = np.concatenate((self.__trajectory,pos),axis=0)

    @property
    def car(self):
        return self.__car


class Avoid_Soccor_Env(object):
    def __init__(self,state_dim = [5,360],action_dim = 1,a_info = 'angle',reset = False,seed = 0,batch_file = -1):
        self.viewer = None
        self.__reset = reset

        """ set seed """
        np.random.seed(seed)
        
        """ state """
        self.pos_dim = state_dim[0]
        self.scan_dim = state_dim[1]
        
        """ action """
        self.action_dim = action_dim
        self.a_info = a_info

        """ log """
        self.step_log = 0.0
        self.goal_car_dis_log = 0.0

        """ element """
        self.maze = MAZE()
        self.robot = Car(scan_num = self.scan_dim)

        """ obstacle limit """
        self.ob_range_x = [-200,200]
        self.ob_range_y = [-177,177]
        self.ob_range_yaw = [-180,180]

        """ robot range limit """
        self.range_x = [-300,310]
        self.range_y = [-200,200]
        self.range_yaw = [-180,180]

        """ soccer filed range limit """
        if(batch_file >= 0):
            path = rospkg.RosPack().get_path('nubot_strategy')+\
                   '/avoid_figure/lib/train_batch/num_{}/'.format(self.maze.ob_num)
            self.train_data = self.Read_Train_Data(path+'{}_env.pickle'.format(batch_file))

        """ viewer range limit """
        self.v_range_x = [-self.maze.width/2,self.maze.width/2] 
        self.v_range_y = [-self.maze.height/2,self.maze.height/2]

        self.Init_Pos(reset)

    def Init_Pos(self,reset = 3):
        """ reset
            1:all random
            2:random obstacle and robot
            0:random obstacle
        """
        models = []
        if(reset == 7):
            """ obstacle """
            idx = np.random.randint(7777)%len(self.train_data)
            data = np.array(self.train_data[idx],dtype='int32')*100
            
            for i in range(self.maze.ob_num):
                self.maze.obstacles[i]['pos'] = [*data[i]]
                models.append([*data[i],self.maze.obstacles[i]['range'][0]])
                self.Set_Model_Pos(np.array(data[i]),0,self.maze.obstacles[i]['name'])
            
            """ robot """
            while True:
                x,y,ang = self.Random_Pose(False)
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    models.append([x,y,self.robot.car['radius']])
                    self.Set_Model_Pos(np.array([x,y]),self.robot.car['ang'],self.robot.car['name'])
                    self.robot.Set_Car_Init_Pos(np.array([x,y]))
                    break
            """ goal """
            while True:
                x,y,ang = self.Random_Pose(False)
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    models.append([x,y,self.robot.car['radius']])
                    self.Set_Model_Pos(np.array([x,y]),0,self.maze.goal['name'],np.array([20,20]))
                    break
        elif(reset == 3):
            """ robot """
            x,y,ang = self.Random_Pose(False)
            # print('robot',x,y,0)
            models.append([x,y,self.robot.car['radius']])
            self.Set_Model_Pos(np.array([x,y]),self.robot.car['ang'],self.robot.car['name'])
            self.robot.Set_Car_Init_Pos(np.array([x,y]))
            """ goal """
            x,y,ang = self.Random_Pose(False)
            # print('goal',x,y,0)
            models.append([x,y,self.robot.car['radius']])
            self.Set_Model_Pos(np.array([x,y]),0,self.maze.goal['name'],np.array([20,20]))
            
            print('robot {} goal {}'.format(self.robot.car['pos'],self.maze.goal['pos']))
            """ obstacle """
            i = 0
            while i < len(self.maze.obstacles):
                x,y,ang = self.Random_Pose()
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    # print('obstacle',i,x,y,0)
                    models.append([x,y,self.maze.obstacles[i]['range'][0]])
                    self.Set_Model_Pos(np.array([x,y]),0,self.maze.obstacles[i]['name'])
                    i += 1
        elif(reset == 2):
            """ robot """
            x,y,ang = self.Random_Pose(False)
            models.append([x,y,self.robot.car['radius']])
            self.Set_Model_Pos(np.array([x,y]),self.robot.car['ang'],self.robot.car['name'])
            self.robot.Set_Car_Init_Pos(np.array([x,y]))
            """ goal """
            self.maze.Set_Goal_Pos(self.maze.goal['o_pos'],0)
            self.Set_Model_Pos(self.maze.goal['o_pos'],0,self.maze.goal['name'])

            """ obstacle """
            i = 0
            while i < len(self.maze.obstacles):
                x,y,ang = self.Random_Pose()
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    # print('obstacle',i,x,y,0)
                    models.append([x,y,self.maze.obstacles[i]['range'][0]])
                    self.Set_Model_Pos(np.array([x,y]),0,self.maze.obstacles[i]['name'])
                    i += 1
        elif(reset == 1):
            """ robot """
            self.Set_Model_Pos(self.robot.car['o_pos'],self.robot.car['o_ang'],self.robot.car['name'])
            self.robot.Set_Car_Init_Pos(np.array([*self.robot.car['o_pos']]))
            """ goal """
            self.maze.Set_Goal_Pos(self.maze.goal['o_pos'],0)
            self.Set_Model_Pos(self.maze.goal['o_pos'],0,self.maze.goal['name'])

            """ obstacle """
            i = 0
            while i < len(self.maze.obstacles):
                x,y,ang = self.Random_Pose()
                if(self.Check_Model_Overlapping([x,y],models) == False):
                    # print('obstacle',i,x,y,0)
                    models.append([x,y,self.maze.obstacles[i]['range'][0]])
                    self.Set_Model_Pos(np.array([x,y]),0,self.maze.obstacles[i]['name'])
                    i += 1
        elif(reset == 0):
            """ robot """
            self.Set_Model_Pos(self.robot.car['init_pos'],self.robot.car['o_ang'],self.robot.car['name'])
            print('robot {} goal {}'.format(self.robot.car['init_pos'],self.maze.goal['pos']))
    def Get_Scan(self):
        scan = np.ones(self.robot.car['scan_num'])*self.robot.car['scan_dis']
        obs_idx = np.zeros(len(self.maze.obstacles))
        ang_range = np.linspace(-math.pi,math.pi,self.robot.car['scan_num'],endpoint=False)

        for i in range(len(obs_idx)):
            ang = math.atan2(self.maze.obstacles[i]['pos'][1]-self.robot.car['pos'][1],\
                             self.maze.obstacles[i]['pos'][0]-self.robot.car['pos'][0])
            ang = self.Norm_Angle(math.degrees(ang-math.pi))
            if(ang < 0):
                ang += 360
            obs_idx[i] = int(ang/(360/self.robot.car['scan_num']))
        
        """ detect """
        length = math.ceil(len(scan)/2)-1
        for idx,i in enumerate(obs_idx):
            f_flag = 0
            b_flag = 0
            car_ = None
            for j in range(length):
                car_ = copy.deepcopy(Pos_Soccer2Draw(self.robot.car['pos']))
                if(f_flag == 0):
                    front = int((i+j)%len(scan))
                    f_car = copy.deepcopy(car_)
                if(b_flag == 0):
                    back = int((i-j)%len(scan))
                    b_car = copy.deepcopy(car_)

                for k in np.linspace(1,self.robot.car['scan_dis'],self.robot.car['scan_dis']):
                    if(f_flag == 0):
                        f_car[0] = (car_[0] + k*math.cos(ang_range[front]))
                        f_car[1] = (car_[1] + k*math.sin(ang_range[front]))
                        if(k == self.robot.car['scan_dis']):
                            f_flag = 1
                        if(self.Check_Obstacle(f_car,self.maze.obstacles[idx]['pos'],self.maze.obstacles[idx]['range'])):
                            if(k < scan[front]):
                                scan[front] = k
                            break
                    else:
                        break
                for k in np.linspace(1,self.robot.car['scan_dis'],self.robot.car['scan_dis']):
                    if(b_flag == 0):
                        b_car[0] = (car_[0] + k*math.cos(ang_range[back]))
                        b_car[1] = (car_[1] + k*math.sin(ang_range[back]))
                        if(k == self.robot.car['scan_dis']):
                            b_flag = 1
                        if(self.Check_Obstacle(b_car,self.maze.obstacles[idx]['pos'],self.maze.obstacles[idx]['range'])):
                            if(k < scan[back]):
                                scan[back] = k
                            break
                    else:
                        break
                if(f_flag == 1 and b_flag == 1):
                    break
        return scan
    
    def Get_Obs(self):
        if(self.pos_dim == 5):
            """ goal pos """
            # p_robot = np.concatenate((self.robot.car['pos'],np.array([self.robot.car['ang']])),axis=0)
            # pos = np.concatenate((p_robot,self.maze.goal['pos']),axis=0)
            # obs = np.hstack((pos,self.Get_Scan()))
            # obs /= 100.0

            """ dis angle """
            r_pos = np.round(self.robot.car['pos']/100.0,2)
            r_ang = np.array([math.radians(self.robot.car['ang'])])
            p_robot = np.concatenate((r_pos,r_ang),axis=0)

            dis = self.Cal_Goal_Robot_Dis()/100.0

            ang = self.Cal_Goal_Robot_Ang('rad')

            vec = np.concatenate((p_robot,np.array([dis,ang])),axis=0)
            obs = np.hstack((vec,self.Get_Scan()/100.0))
            
            # print('ang : ',math.degrees(ang))
            # print(p_robot)
            # print(obs[:5])
        elif(self.pos_dim == 2):
            """ move vector """
            move = self.maze.goal['pos'] - self.robot.car['pos']
            obs = np.hstack((move,self.Get_Scan()))
            obs /= 100.0

        # print(obs)
        return obs
    
    def Is_Done(self,obs):
        """ bump """
        dis_o = min(obs[self.pos_dim:])
        if(dis_o <= (self.robot.car['radius']+2)/100.0):
            return True,'bump'
            # return False,'bump'
        """ over range """
        if(self.Check_Robot_Over_Range()):
            return True,'over range'
            # return False,'over range'
        
        """ goal """
        if(self.Check_Arrive_GoalAera()):
            return True,'goal'
            # return False,'goal'

        return False,'None'
    
    def Compute_Reward(self,obs,info,action):

        if(self.a_info == 'angle'):
            angle = action[0]
        elif(self.a_info == 'linear'):
            angle = math.atan2(action[1],action[0])
            v = math.hypot(*action[:2])

        # dis for meter
        dis = self.Cal_Goal_Robot_Dis()/100.0

        """ reward 1 """
        # scalar = 1.0
        # if((self.goal_car_dis_log - dis) >= 0):
        #     reward = scalar*abs(6-dis)
        # else:
        #     reward = -scalar*abs(dis)

        """ reward 2 """
        # reward = -dis

        """ reward 3 """
        reward = -dis/40.0
        reward += round(4.76*(1.05-abs(obs[4]-angle)),2)
        
        if(info == 'goal'):
            reward += 100.
        elif(info == 'bump'):
            reward -= 500.
        elif(info == 'over range'):
            reward -= 500.
        
        return reward,info
        """ reward 4 """
        # dis_o = min(obs[self.pos_dim:])
        # x1 = abs(self.range_x[0]-self.robot.car['pos'][0])
        # x2 = abs(self.range_x[1]-self.robot.car['pos'][0])
        # y1 = abs(self.range_y[0]-self.robot.car['pos'][1])
        # y2 = abs(self.range_y[1]-self.robot.car['pos'][1])
        # dis_f = round(min([x1,x2,y1,y2]),3)/100.0
        # bad_dis = min([dis_o,dis_f])

        # reward = (-dis+0.5)/10
        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward += (cos_vec*dis - dis)/10
        
        # if(reward > 0):
        #     reward *= 3

        # self.goal_car_dis_log = dis
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

        """ reward 5 """
        # reward = -dis+0.3
        # cos_vec = np.cos(abs(obs[4]-angle))
        # reward += (cos_vec*0.2-0.01)
        
        # if(info == 'goal'):
        #     reward = 10.
        # elif(info == 'bump'):
        #     reward = -10.
        # elif(info == 'over range'):
        #     reward = -10.
        # return reward,info
        """ reward 6 """
        # reward = -0.01
        # if(info == 'goal'):
        #     reward = 10.
        # elif(info == 'bump'):
        #     reward = -10.
        # elif(info == 'over range'):
        #     reward = -10.

        # return reward,info

    def Step(self,action):
        if(self.a_info == 'angle'):
            x = self.robot.car['pos'][0]+self.robot.car['vec']*math.cos(math.radians(self.robot.car['ang'])+action)
            y = self.robot.car['pos'][1]+self.robot.car['vec']*math.sin(math.radians(self.robot.car['ang'])+action)
        elif(self.a_info == 'linear'):
            x = self.robot.car['pos'][0] + int(action[0])
            y = self.robot.car['pos'][1] + int(action[1])
            
        x = np.clip(x,*self.v_range_x)
        y = np.clip(y,*self.v_range_y)
        
        self.Set_Model_Pos(np.array([x,y]),self.robot.car['ang'],self.robot.car['name'])
        self.robot.Set_Car_Trajectory([x,y])

        """ state """
        obs = self.Get_Obs()

        """ done """
        done,info = self.Is_Done(obs)

        """ reward """
        reward,info = self.Compute_Reward(obs,info,action)

        return obs,reward,done,info
    
    def Reset(self,reset = 1):
        """ log """
        self.step_log = 0.0
    
        self.Init_Pos(reset)
        self.goal_car_dis_log = self.Cal_Goal_Robot_Dis()/100.0
        return self.Get_Obs()
    
    def Stop_Env(self):
        pass

    def render(self):
        if(self.viewer is None):
            self.viewer = Viewer(scan_num = self.scan_dim)
            """ robot """
            self.Set_Model_Pos(self.robot.car['pos'],\
                               self.robot.car['ang'],\
                               self.robot.car['name'])

            """ scan """
            self.viewer.Set_Scan(self.Get_Scan())

            """ goal """
            self.Set_Model_Pos(self.maze.goal['pos'],\
                               self.maze.goal['ang'],\
                               self.maze.goal['name'],\
                               self.maze.goal['range'])

            """ obstacle """
            for item in self.maze.obstacles:
                self.Set_Model_Pos(item['pos'],\
                                   item['ang'],\
                                   item['name'])

        self.viewer.render()
    
    
    """ set model """
    def Set_Model_Pos(self,pos,ang,model,goal_range = None):
        idx = model.find('_')
        if(idx != -1):
            self.maze.Set_Obstacle_Pos(int(model[-1]),pos,ang)
        else:
            if(model == self.robot.car['name']):
                self.robot.Set_Car_Pos(pos,ang)
                if(self.viewer is not None):
                    self.viewer.Set_Scan(self.Get_Scan())
            elif(model == self.maze.goal['name']):
                self.maze.Set_Goal_Pos(pos,ang,goal_range)
        if(self.viewer is not None):
            self.viewer.Set_Model_Pos(pos,ang,model,goal_range)
            
    """ tool """
    def Random_Pose(self,obstacle = True,ob_error = 1,r_error = 25):
        if(obstacle):
            x = np.random.uniform(self.ob_range_x[0]+ob_error,self.ob_range_x[1]-ob_error,1)
            y = np.random.uniform(self.ob_range_y[0]+ob_error,self.ob_range_y[1]-ob_error,1)
            angle = np.random.uniform(*self.ob_range_yaw,1)
        else:
            x = np.random.uniform(self.range_x[0]+r_error,self.range_x[1]-r_error,1)
            y = np.random.uniform(self.range_y[0]+r_error,self.range_y[1]-r_error,1)
            angle = np.random.uniform(*self.range_yaw,1)
        return round(x[0]),round(y[0]),round(angle[0])
    
    """ bug """
    def Check_Model_Overlapping(self,pos,models,error = 70):
        """ method 2 """
        for item in models:
            if(math.hypot(pos[0]-item[0],pos[1]-item[1]) < error):
                return True
        return False

    
    def Check_Obstacle(self,pos,obstacle,size):
        pos1 = pos
        pos2 = Pos_Soccer2Draw(obstacle)
        
        if(pos1[0] < (pos2[0]-size[0])):
            return False
        if(pos1[0] > (pos2[0]+size[0])):
            return False
        if(pos1[1] < (pos2[1]-size[1])):
            return False
        if(pos1[1] > (pos2[1]+size[1])):
            return False
        return True
        
    def Check_Arrive_GoalAera(self):
        
        range_x1 = (self.maze.goal['pos'][0]-self.maze.goal['range'][0])
        range_x2 = (self.maze.goal['pos'][0]+self.maze.goal['range'][0])
        range_y1 = (self.maze.goal['pos'][1]-self.maze.goal['range'][1])
        range_y2 = (self.maze.goal['pos'][1]+self.maze.goal['range'][1])

        if(self.robot.car['pos'][0] < range_x1):
            return False
        if(self.robot.car['pos'][0] > range_x2):
            return False
        if(self.robot.car['pos'][1] < range_y1):
            return False
        if(self.robot.car['pos'][1] > range_y2):
            return False
        return True
    
    def Check_Robot_Over_Range(self):
        if(self.robot.car['pos'][0] < self.range_x[0]):
            return True
        if(self.robot.car['pos'][0] > self.range_x[1]):
            return True
        if(self.robot.car['pos'][1] < self.range_y[0]):
            return True
        if(self.robot.car['pos'][1] > self.range_y[1]):
            return True
        return False
    
    def Cal_Goal_Robot_Dis(self):
        return np.hypot(*(self.robot.car['pos'] - self.maze.goal['pos']))
    
    def Cal_Goal_Robot_Ang(self,ang_rad = 'deg'):
        dx = self.maze.goal['pos'][0] - self.robot.car['pos'][0]
        dy = self.maze.goal['pos'][1] - self.robot.car['pos'][1]
        ang = math.degrees(math.atan2(dy,dx))
        ang -= self.robot.car['ang']
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
    
    def Read_Train_Data(self,path):
        with open(path,'rb') as f:
            data = pickle.load(f)
            return data


class Viewer(pyglet.window.Window):
    def __init__(self,scan_num = 20):
        self.maze = MAZE()
        self.robot = Car(scan_num = scan_num)

        super(Viewer, self).__init__(width=self.maze.width, height=self.maze.height, resizable=True, caption='soccer_car', vsync=False)
        
        """ clear """
        pyglet.gl.glClearColor(1, 1, 1, 1)

        """ scan """
        self.scan = [0,0]*self.robot.car['scan_num']

        """ init object """
        self.batch = pyglet.graphics.Batch()
        self.robot_batch = pyglet.graphics.Batch()
        self.Init_SimGoal_And_SimField()
        self.Init_SimObstacle()
        self.Init_SimRobot()
    
    def Init_SimGoal_And_SimField(self):
        goal_pos = Pos_Soccer2Draw(self.maze.goal['pos'])
        goal_v = np.hstack([goal_pos-self.maze.goal['range'],\
                            goal_pos[0]-self.maze.goal['range'][0],\
                            goal_pos[1]+self.maze.goal['range'][1],\
                            goal_pos+self.maze.goal['range'],\
                            goal_pos[0]+self.maze.goal['range'][0],\
                            goal_pos[1]-self.maze.goal['range'][1]])

        self.simGoal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', goal_v),
            ('c3B', (86, 109, 249) * 4))    # color

        field_pos = np.array([350,250])
        field_range = np.array([300,200])
        field_v = np.hstack([field_pos-field_range,\
                             field_pos[0]-field_range[0],\
                             field_pos[1]+field_range[1],\
                             field_pos+field_range,\
                             field_pos[0]+field_range[0],\
                             field_pos[1]-field_range[1]])

        self.simField = self.batch.add(
            4, pyglet.gl.GL_LINE_LOOP, None,    # 4 corners
            ('v2f', field_v),
            ('c3B', (0, 0, 0) * 4))    # color
    
    def Init_SimObstacle(self):
        obstacles_v = []
        for item in self.maze.obstacles:
            item_pos = Pos_Soccer2Draw(item['pos'])
            obstacles_v.append(np.hstack([item_pos-item['range'],\
                                          item_pos[0]-item['range'][0],\
                                          item_pos[1]+item['range'][1],\
                                          item_pos+item['range'],\
                                          item_pos[0]+item['range'][0],\
                                          item_pos[1]-item['range'][1]]))
        
        self.simObstacle = []
        for item in obstacles_v:
            self.simObstacle.append(self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', item),
            ('c3B', (249, 86, 86) * 4,)))
    
    def Init_SimRobot(self):
        """ robot """
        car_pos = Pos_Soccer2Draw(self.robot.car['pos'])
        robot_v = self.Make_Circle(360,self.robot.car['radius'],*car_pos)
        self.simCar = self.robot_batch.add(
            int(len(robot_v)/2), pyglet.gl.GL_LINE_LOOP, None,
            ('v2f', robot_v), ('c3B', (133, 133, 133) * int(len(robot_v)/2)))
        
        """ front """
        robotF_v = self.Make_Front(self.robot.car['ang'],self.robot.car['radius'],*car_pos)
        self.simCarF = self.robot_batch.add(
            2, pyglet.gl.GL_LINES, None,
            ('v2f', robotF_v), ('c3B', (133, 133, 133) * 2))

        """ scan """
        """ rgb(224,137,31) """
        scan_v = self.Make_Scan()
        self.simScan = self.batch.add(
            int(len(scan_v)/2), pyglet.gl.GL_LINES, None,
            ('v2f', scan_v), ('c3B', (224,137,31) * int(len(scan_v)/2)))
        
    """ set model """
    def Set_Model_Pos(self,pos,ang,model,goal_range = None):
        idx = model.find('_')
        if(idx != -1):
            self.maze.Set_Obstacle_Pos(int(model[-1]),pos,ang)
        else:
            if(model == self.robot.car['name']):
                self.robot.Set_Car_Pos(pos,ang)
            elif(model == self.maze.goal['name']):
                self.maze.Set_Goal_Pos(pos,ang,goal_range)

    def Set_Scan(self,scan):
        self.scan = scan 
    
    """ update """
    def update(self):
        """ obstacle """
        i = 0
        for item in self.maze.obstacles:
            item_pos = Pos_Soccer2Draw(item['pos'])
            obstacles_v = np.hstack([item_pos-item['range'],\
                                     item_pos[0]-item['range'][0],\
                                     item_pos[1]+item['range'][1],\
                                     item_pos+item['range'],\
                                     item_pos[0]+item['range'][0],\
                                     item_pos[1]-item['range'][1]])
            self.simObstacle[i].vertices = obstacles_v
            i += 1
        
        """ goal """
        goal_pos = Pos_Soccer2Draw(self.maze.goal['pos'])
        goal_v = np.hstack([goal_pos-self.maze.goal['range'],\
                            goal_pos[0]-self.maze.goal['range'][0],\
                            goal_pos[1]+self.maze.goal['range'][1],\
                            goal_pos+self.maze.goal['range'],\
                            goal_pos[0]+self.maze.goal['range'][0],\
                            goal_pos[1]-self.maze.goal['range'][1]])
        self.simGoal.vertices = goal_v
        """ car """
        car_pos = Pos_Soccer2Draw(self.robot.car['pos'])
        robot_v = self.Make_Circle(360,self.robot.car['radius'],*car_pos)
        self.simCar.vertices = robot_v

        robotF_v = self.Make_Front(self.robot.car['ang'],self.robot.car['radius'],*car_pos)
        self.simCarF.vertices = robotF_v

        scan_v = self.Make_Scan()
        self.simScan.vertices = scan_v

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.robot_batch.draw()
    
    def on_close(self):
        self.close()

    """ view """
    def render(self):
        self.update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()
    
    """ tool """
    
    def Make_Circle(self,numPoints,r,c_x,c_y):
        verts = []
        for i in range(numPoints):
            angle = math.radians(float(i)/numPoints * 360.0)
            x = r*math.cos(angle) + c_x
            y = r*math.sin(angle) + c_y
            verts += [x,y]
        return verts

    def Make_Front(self,ang,r,c_x,c_y):
        angle = math.radians(ang)
        x = r*math.cos(angle) + c_x
        y = r*math.sin(angle) + c_y

        return [x,y,c_x,c_y]
    
    def Make_Scan(self):
        scan_v = []
        pos = Pos_Soccer2Draw(self.robot.car['pos'])
        for i,j in zip(np.linspace(-math.pi,math.pi,self.robot.car['scan_num'],endpoint=False),range(self.robot.car['scan_num'])):
            scan_v.append(pos)
            x = pos[0] + (self.scan[j])*math.cos(i)
            y = pos[1] + (self.scan[j])*math.sin(i)
            scan_v.append([x,y])
        return np.hstack(scan_v)


###################################################################
def main():
    # env = Viewer()
    
    env = Avoid_Soccor_Env(reset = True)
    env.Reset()
    while True:
        env.render()
        obs,reward,done,info = env.Step(np.array([0]))
        if(done == True):
            env.Reset()

if __name__ == '__main__':
    main()
