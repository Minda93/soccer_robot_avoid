#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" ros """
import rospy
from std_srvs.srv import Empty

from gazebo_msgs.msg import ContactsState,ContactState
from sensor_msgs.msg import LaserScan
from nubot_common.msg import VelCmd
from nubot_strategy.msg import reward
from nubot_strategy.msg import tutor

""" lib """
from lib.tool.utils import ReplayBuffer_Deque

""" tool """
import math
import numpy as np

SCAN_LIMIT = 2.5
BUMP_LIMIT = 0.26

class SimNodeHandle(object):
    def __init__(self,robot_name,scan_dim,a_info):
        self.__robot = robot_name

        self.__bumper = False
        self.__scan = None
        self.scan_dim = scan_dim
        self.a_info = a_info

        self.buffer = []

        """ pub """
        self.pub_cmdvel = rospy.Publisher(robot_name+'/nubotcontrol/velcmd',VelCmd, queue_size = 1)
        self.pub_reward = rospy.Publisher('/reward',reward, queue_size = 1)

        """ sub """
        rospy.Subscriber("/bumper",ContactsState,self.Sub_Bumper)
        rospy.Subscriber("/scan",LaserScan,self.Sub_Scan)
        rospy.Subscriber('/tutor',tutor,self.Sub_Tutor)
        """ service """
        self.reset_dynamic = rospy.ServiceProxy('/dynamic_env/reset', Empty)

        print("sim nh init")

    def Calculate_Vel(self,action):
        x = action[0]*math.cos(action[1])
        y = action[0]*math.sin(action[1])
        return x,y

    """ publish """
    def Pub_Cmdvel(self,vec,action_dim = 1):
        velcmd = VelCmd()

        if(self.a_info == 'angle'):
            velcmd.Vx,velcmd.Vy = self.Calculate_Vel(vec[:2])
            velcmd.w = vec[2]
        elif(self.a_info == 'scalar_ang'):
            velcmd.Vx,velcmd.Vy = self.Calculate_Vel(vec[:2])
            velcmd.w = vec[2]
        elif(self.a_info == 'linear'):
            velcmd.Vx = vec[0]
            velcmd.Vy = vec[1]
            velcmd.w = 0
        elif(self.a_info == 'linear_yaw'):
            velcmd.Vx = vec[0]
            velcmd.Vy = vec[1]
            velcmd.w = vec[2]

        self.pub_cmdvel.publish(velcmd)
    
    def Pub_Info(self,episode,step,r,info):
        msg = reward()
        msg.episode = episode
        msg.step = step
        msg.reward = r
        msg.info = info
        self.pub_reward.publish(msg)
    
    """ subscribe """
    def Sub_Bumper(self,msg = None):
        while msg is None and not rospy.is_shutdown():
            try:
                # msg = rospy.wait_for_message("/bumper", ContactsState)
                msg = rospy.wait_for_message("/bumper", ContactsState, timeout=1.)
            except:
                print("don't listen bumper")

        if(len(msg.states)):
            for state in msg.states:
                if(self.__robot in state.info):
                    self.__bumper = True

    def Sub_Scan(self,msg = None):
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/scan", LaserScan)
                # msg = rospy.wait_for_message("/scan", LaserScan, timeout=5.)
            except:
                print("don't listen scan")
        
        scan = []
        for i in range(len(msg.ranges)):
            if((i%(360/self.scan_dim)) == 0):
                if(msg.ranges[i] == math.inf):
                    scan.append(SCAN_LIMIT)
                elif(msg.ranges[i] == math.isnan):
                    scan.append(0.0)
                else:
                    # scan.append(msg.ranges[i])
                    scan.append(round(msg.ranges[i],2))

        """ 270 """
        # scan = []
        # for i in range(len(msg.ranges)):
        #     if(i <= 135):
        #         if(msg.ranges[i] == math.inf):
        #             scan.append(SCAN_LIMIT)
        #         elif(msg.ranges[i] == math.isnan):
        #             scan.append(0.0)
        #         else:
        #             scan.append(round(msg.ranges[i],3))
        #     if(i > 225):
        #         if(msg.ranges[i] == math.inf):
        #             scan.append(SCAN_LIMIT)
        #         elif(msg.ranges[i] == math.isnan):
        #             scan.append(0.0)
        #         else:
        #             scan.append(round(msg.ranges[i],3))

        self.__scan = np.array(scan)
        if(np.amin(self.__scan) <= BUMP_LIMIT):
            self.__bumper = True
    
    def Sub_Tutor(self,msg):
        buffer = []
        buffer.append(msg.state)
        buffer.append(msg.action)
        buffer.append(msg.reward)
        buffer.append(msg.new_state)
        buffer.append(msg.done)
        self.buffer.append(buffer)
    
    def Reset_Bumper(self):
        self.__bumper = False
    
    def Reset_Dynamic_Env(self):
        rospy.wait_for_service('/dynamic_env/reset')
        try:
            self.reset_dynamic()
        except rospy.ServiceException as e:
            print ("/dynamic_env/reset service call failed")
    
    def Store_Buffer(self,buffer):
        for i in range(len(self.buffer)):
            if(self.a_info == 'angle'):
                buffer.Add(self.buffer[i][0],\
                           np.reshape(self.buffer[i][1],(1,)),\
                           self.buffer[i][2],\
                           self.buffer[i][3],\
                           self.buffer[i][4])
        self.buffer = []

    """ check topic """
    def Check_Connection(self):
        init = None
        while init is None and not rospy.is_shutdown():
            try:
                init = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
            except:
                print('scan not init')
        
        init = None
        while init is None and not rospy.is_shutdown():
            try:
                init = rospy.wait_for_message("/bumper", ContactsState, timeout=1.0)
            except:
                print('bumper not init')

        # print('init success nh')
        return True

    @property
    def scan(self):
        return self.__scan
    
    @property
    def bumper(self):
        return self.__bumper