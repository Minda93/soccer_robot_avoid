#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

""" ros """
import rospy
from gazebo_msgs.msg import ModelStates,ModelState

""" lib """
from lib.gazebo.gazebo_connection import GazeboConnection
from lib.tool.counter import TimeCounter
from lib.tool.euler_orientation import Get_Z_Euler,Get_Orientation

""" tool """
from numpy import random
import time
import math

class RobotGazeboEnv(GazeboConnection):

    def __init__(self,robot_name,reset_world_or_sim = "SIMULATION"):
        super(RobotGazeboEnv,self).__init__(
            reset_world_or_sim = reset_world_or_sim
        )

        """ model states """
        self.modelStates = {}

        """ robot """
        self.__robotName = robot_name
        self.robotPos = [0.0,0.0,0.0]
        self.robotVec = [0.0,0.0]

        """ init """
        self.Check_Model_States()

        """ pub """
        self.pub_setModel = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        """ sub """
        rospy.Subscriber("gazebo/model_states",ModelStates,self.Sub_Robot_Model)

        print("gazebo env init")

    def Check_Model_States(self):
        self.Start_Sim()
        
        msg = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
            except:
                print('model states not init')
       
        self.Init_Model_States(msg)
    
    def Init_Model_States(self,msg):
        i = 0
        self.t1 = time.time()
        for model in msg.name:
            state = {}
            state['pose'] = msg.pose[i]
            state['twist'] = [msg.twist[i].linear.x,msg.twist[i].linear.y,msg.twist[i].angular.z]
            self.modelStates[model] = state

            if(msg.name[i] == self.__robotName):
                self.robotPos[0] = round(msg.pose[i].position.x,3)
                self.robotPos[1] = round(msg.pose[i].position.y,3)
                q = msg.pose[i].orientation
                self.robotPos[2] = Get_Z_Euler(q.x,q.y,q.z,q.w)

                front = math.radians(self.robotPos[2])
                x = round(msg.twist[i].linear.x,3)
                y = round(msg.twist[i].linear.y,3)
                self.robotVec[0] = math.cos(front)*x - math.sin(front)*y
                self.robotVec[1] = math.sin(front)*x + math.cos(front)*y
            i += 1
    def Sub_Robot_Model(self,msg = None):
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/gazebo/model_states", ModelStates)
            except:
                print("don't listen robot")

        i = 0
        for model in msg.name:
            if(msg.name[i] == self.__robotName):
                self.robotPos[0] = round(msg.pose[i].position.x,3)
                self.robotPos[1] = round(msg.pose[i].position.y,3)
                q = msg.pose[i].orientation
                self.robotPos[2] = Get_Z_Euler(q.x,q.y,q.z,q.w)

                front = math.radians(self.robotPos[2])
                x = round(msg.twist[i].linear.x,3)
                y = round(msg.twist[i].linear.y,3)
                self.robotVec[0] = math.cos(front)*x - math.sin(front)*y
                self.robotVec[1] = math.sin(front)*x + math.cos(front)*y

            i += 1
    
    def Set_Model(self,name,pose,angle):
        if(name in self.modelStates):
            model = ModelState()
            model.model_name = name
            model.pose.position.x = pose[0]
            model.pose.position.y = pose[1]
            model.pose.position.z = 0.1
            qx,qy,qz,qw = Get_Orientation(0.0,0.0,angle)
            model.pose.orientation.x = qx
            model.pose.orientation.y = qy
            model.pose.orientation.z = qz
            model.pose.orientation.w = qw

            for i in range(5):
                while True:
                    self.pub_setModel.publish(model)
                    self.Check_Model_States()
                    if(self.Check_Model_Pose(name,pose,qx,qy,qz,qw)):
                        # print("model setup success")
                        break

    def Check_Model_Pose(self,name,pose,qx,qy,qz,qw):
        error = 0.01
        if(name in self.modelStates):
            if(abs(pose[0]-round(self.modelStates[name]['pose'].position.x,2)) > error):
                return False
            if(abs(pose[1]-round(self.modelStates[name]['pose'].position.y,2)) > error):
                return False
            # if(abs(pose[2]-round(self.modelStates[name]['pose'].position.z,2)) > error):
            #     return False
            
            if(abs(qx-round(self.modelStates[name]['pose'].orientation.x,2)) > error):
                return False
            if(abs(qy-round(self.modelStates[name]['pose'].orientation.y,2)) > error):
                return False
            if(abs(qz-round(self.modelStates[name]['pose'].orientation.z,2)) > error):
                return False
            if(abs(qw-round(self.modelStates[name]['pose'].orientation.w,2)) > error):
                return False
            return True
    
    def Check_Robot_Stop(self,times = 2):
        if(self.__robotName in self.modelStates):
            i = 0
            self.Start_Sim()
            while (i < times):
                self._Init_Robot_Control()
                self.Check_Model_States()
                if(all(round(value,2) == 0. for value in self.modelStates[self.__robotName]['twist'])):
                    i += 1
                    self.Start_Sim()
                    break
            return True
        else:
            print('not find robot')
            print('not find robot')
            print('not find robot')
    
    """ Reset Sim """
    r""" reset
        reset = 0 -> no reset
        reset = 1 -> only obstacle random 
        reset = 2 -> randorm robot and obstacle
        reset = 3 -> all random
        reset = 4 -> only goal random
        reset = 5 -> random goal and robot
    """
    def Reset_Sim(self,reset = 1):
        # print("reset SIM Start")
        self.Start_Sim()
        self._Init_Control()
        self._Check_All_System_Ready()

        self.Stop_Sim()
        # self.resetSim()
        self.Start_Sim()
        self._Init_Control()

        # self.resetSim()
        self._Init_Pose(reset)
        self._Init_Dynamic_Env()
        self._Init_Dynamic_Env()
        self._Check_All_System_Ready()
        self.Check_Model_States()
        self.Stop_Sim()

        # print("reset SIM end")

    """ extend method """
    def _Init_Control(self):
        raise NotImplementedError()
        # pass
            
    
    def _Init_Pose(self,reset = True):
        raise NotImplementedError()
        # pass
    
    def _Init_Robot_Control(self,modelName):
        raise NotImplementedError()
        # pass
    
    def _Check_All_System_Ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()
        # pass
    
    def _Init_Dynamic_Env(self):
        raise NotImplementedError()

    """ tool """
    def Norm_Angle(self,angle):
        if(angle > 180):
            angle -=360
        elif(angle < -180):
            angle +=360
        return angle