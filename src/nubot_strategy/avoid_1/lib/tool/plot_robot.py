#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as tf
import pickle

def Read_Data(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
        return data
        

def main():
    file_path = 'sac_7/linear_x_y_no_ra/challenge_fixed/'
    robots = Read_Data(file_path+"log_robot_show.pickle")
    paths = Read_Data(file_path+"log_path_show.pickle")
    obstalces = Read_Data(file_path+"log_obstacle_show.pickle")
    obs_dir = Read_Data("log_direction_show.pickle")
    goals = Read_Data(file_path+"log_goal_show.pickle")
    choise_episode = 3
    print(obs_dir)
    path = paths[choise_episode]
    obstalce = obstalces[choise_episode]
    ob_dir = obs_dir[0]
    goal = goals[choise_episode]

    print(len(paths[0]))
    print(obstalce)
    print(goal)

    """ add plot object """
    ax = plt.gca()

    """ path """
    x = []
    y = []
    for pos in path:
        x.append(pos[0])
        y.append(pos[1])
    plt.plot(x,y,color='blue')

    """ obstacle """
    x = []
    y = []
    for pos in obstalce:
        x.append(pos[0]-0.225)
        y.append(pos[1]-0.225)
    # print(x,y)
    
    for i in range(len(x)):
        center = [x[i]+0.225,y[i]+0.225]

        obstacle = patches.Rectangle((x[i],y[i]), 0.45, 0.45, color='black')
        
        """ obstacle yaw """
        ts = ax.transData
        tr = tf.Affine2D().rotate_deg_around(center[0],center[1], 20)
        t = tr + ts
        obstacle.set_transform(t)

        """ dir """
        if(len(ob_dir)):
            # move_x
            if(ob_dir[i] == 1):
                plt.arrow(center[0],center[1],0.3,0,width=0.04,color='r')
                plt.arrow(center[0],center[1],-0.3,0,width=0.04,color='r')
            # move_y
            elif(ob_dir[i] == 2):
                plt.arrow(center[0],center[1],0,0.3,width=0.04,color='r')
                plt.arrow(center[0],center[1],0,-0.3,width=0.04,color='r')
            # rotate
            elif(ob_dir[i] == 3):
                ax.annotate("",\
                            xy=(center[0], center[1]+0.5),\
                            xycoords='data',\
                            xytext=(center[0]+0.5, center[1]),\
                            textcoords='data',\
                            arrowprops=dict(width=1.5,
                                            headwidth=8,
                                            color='red',
                                            connectionstyle="arc3,rad=0.8"))
        
        ax.add_patch(obstacle)

    """ obstacle dir """
    # straight
    # plt.arrow(0,0,0.25,0,linewidth=5,color='r')
    # plt.arrow(0,0,-0.25,0,linewidth=5,color='r')
    # rotate
    # ax.annotate("",
    #         xy=(0, 0.5), xycoords='data',
    #         xytext=(0.5, 0), textcoords='data',
    #         arrowprops=dict(arrowstyle="<-",
    #                         connectionstyle="arc3,rad=0.8"))

    """ goal """
    # fix goal
    # left buttom point
    y_goal = patches.Rectangle((-3,-0.75), 0.4, 1.5, color='yellow')
    b_goal = patches.Rectangle((2.6,-0.75), 0.4, 1.5, color='blue')
    ax.add_patch(y_goal)
    ax.add_patch(b_goal)

    # random goal
    random_goal = patches.Rectangle((goal[0]-0.22,goal[1]-0.22), 0.44, 0.44, color='red')
    ax.add_patch(random_goal)

    plt.ylim(-2.5, 2.5)
    plt.xlim(-3.5,3.5)
    plt.grid()
    plt.show()
    # plt.savefig('./robot_path1.png')


if __name__ == '__main__':
    main()