#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" ros """
import roslib
roslib.load_manifest('nubot_strategy')
import rospy
import rospkg
from nubot_common.msg import VelCmd
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty,EmptyResponse
""" lib """

""" tool """
import math
import numpy as np
import threading

PACKAGE_PATH = rospkg.RosPack().get_path('nubot_strategy')+'/avoid/'

class Obstacle(object):
    def __init__(self,number,pos,move_way,move_dis = 0.5):
        self.number = number
        self.move_way = move_way
        self.move_dis = move_dis

        self.move = 1
        self.init_pos = np.zeros(2)
        self.pos = np.zeros(2)
        self.goal = np.zeros(2)
        self.arrive = 0

        """ topic name """
        self.topic_vel = "/obstacle_{}/nubotcontrol/velcmd".format(number)

        """ init state """
        self.Init_State(pos)

        """ pub """
        self.pub_cmdvel = rospy.Publisher(self.topic_vel, VelCmd, queue_size=1)

    def Init_State(self,pos):
        self.init_pos[0] = pos[0]
        self.init_pos[1] = pos[1]
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]

        if(self.move_way == 'vertical'):
            x = self.init_pos[0] + self.move*self.move_dis
            y = self.init_pos[1]
            self.Set_Goal(x,y)
        elif(self.move_way == 'horizon'):
            x = self.init_pos[0] 
            y = self.init_pos[1] + self.move*self.move_dis
            self.Set_Goal(x,y)

    def Set_Pos(self,pos):
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]     

    def Set_Goal(self,x,y):
        self.goal[0] = x
        self.goal[1] = y
        # print(x,y)
        
    """ ros """
    def Pub_Cmdvel(self,vec):
        velcmd = VelCmd()
        velcmd.Vx = vec[0]
        velcmd.Vy = vec[1]
        velcmd.w = vec[2]
        self.pub_cmdvel.publish(velcmd)
    
    """ reset """
    def Reset(self,way):
        self.move_way = way
        self.Pub_Cmdvel([0,0,0])

    """ move """
    def Move(self,scalar = 25):
        if(self.move_way == 'stop'):
            vec = [0,0,0]
        elif(self.move_way == 'vertical'):
            vec = self.Move_Vertical(scalar)
        elif(self.move_way == 'horizon'):
            vec = self.Move_Horizon(scalar)
        elif(self.move_way == 'rotate'):
            if(scalar > 10):
                scalar = 10
            vec = [0,0,scalar]
        
        self.Pub_Cmdvel(vec)
    
    def Move_Vertical(self,scalar):

        if(self.move > 0 and self.pos[0] > self.goal[0]):
            x = self.init_pos[0] - 1*self.move_dis
            y = self.init_pos[1] 
            self.Set_Goal(x,y)
        elif(self.move < 0 and self.pos[0] < self.goal[0]):
            x = self.init_pos[0] + 1*self.move_dis
            y = self.init_pos[1] 
            self.Set_Goal(x,y)
        
        if(self.pos[0] > self.goal[0]):
            self.move = 1
        elif(self.pos[0] < self.goal[0]):
            self.move = -1
        
        vx = self.move*scalar
        vy = 0
        w = 0

        return [vx,vy,w]
    
    def Move_Horizon(self,scalar):
        
        if(self.move > 0 and self.pos[1] > self.goal[1]):
            x = self.init_pos[0] 
            y = self.init_pos[1] - 1*self.move_dis
            self.Set_Goal(x,y)
        elif(self.move < 0 and self.pos[1] < self.goal[1]):
            self.move = 1
            x = self.init_pos[0] 
            y = self.init_pos[1] + 1*self.move_dis
            self.Set_Goal(x,y)
        
        if(self.pos[1] > self.goal[1]):
            self.move = 1
        elif(self.pos[1] < self.goal[1]):
            self.move = -1

        vx = 0
        vy = self.move*scalar
        w = 0

        return [vx,vy,w]

    """ tool """
    def Cal_Dis(self):
        return np.hypot(*(self.init_pos - self.pos))
    
class Env(object):
    def __init__(self,num,seed = 2):
        np.random.seed(seed)

        """ obstacle """
        self.obstacle_num = num
        self.obstacles = {}
        self.way = ['stop','vertical','horizon','rotate','vec']
        
        for i in range(1,num+1):
            name = 'obstacle_{}'.format(i)
            choose = np.random.randint(3213)%4
            self.obstacles[name] = Obstacle(i,[0,0],self.way[choose])
        
        self.Init_Env()

        """ sub """
        rospy.Subscriber("gazebo/model_states",ModelStates,self.Sub_Info)

        """ server """
        self.reset = rospy.Service('dynamic_env/reset', Empty, self.Reset)
    
    def Reset(self,req):
        for i in range(1,self.obstacle_num+1):
            name = 'obstacle_{}'.format(i)
            choose = np.random.randint(3213)%4
            self.obstacles[name].Reset(self.way[choose])
        self.Init_Env()
        
        return EmptyResponse()

    def Init_Env(self):
        msg = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
            except:
                print('model states not init')
        i = 0
        for model in msg.name:
            if(model in self.obstacles):
                x = msg.pose[i].position.x
                y = msg.pose[i].position.y
                self.obstacles[model].Init_State([x,y])
            i += 1 
            
    def Sub_Info(self,msg):
        i = 0
        for model in msg.name:
            if(model in self.obstacles):
                x = msg.pose[i].position.x
                y = msg.pose[i].position.y
                self.obstacles[model].Set_Pos([x,y])
            i += 1
    
    def Start(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            for key in self.obstacles.keys():
                self.obstacles[key].Move()
            rate.sleep()

def main():
    rospy.init_node('dynamic_env', anonymous=True)
    env = Env(num = 5,seed = 0)

    env.Start()


if __name__ == '__main__':
    main()
