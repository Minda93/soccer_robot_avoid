#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

def Read_Data(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
        return data
        

def main():
    file_path = 'fix_five_fix_linear/'
    robots = Read_Data(file_path+"log_robot_show.pickle")
    paths = Read_Data(file_path+"log_path_show.pickle")
    obstalces = Read_Data(file_path+"log_obstacle_show.pickle")
    goals = Read_Data(file_path+"log_goal_show.pickle")
    choise_episode = 19
    
    path = paths[choise_episode]
    obstalce = obstalces[choise_episode]
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
        obstacle = patches.Rectangle((x[i],y[i]), 0.45, 0.45, color='black')
        ax.add_patch(obstacle)

    """ goal """
    # fix goal
    # left buttom point
    # y_goal = patches.Rectangle((-3,-0.75), 0.4, 1.5, color='yellow')
    # b_goal = patches.Rectangle((2.6,-0.75), 0.4, 1.5, color='blue')
    # ax.add_patch(y_goal)
    # ax.add_patch(b_goal)

    # random goal
    random_goal = patches.Rectangle((goal[0]-0.22,goal[1]-0.22), 0.44, 0.44, color='red')
    ax.add_patch(random_goal)

    plt.ylim(-2.5, 2.5)
    plt.xlim(-3.5,3.5)
    plt.show()
    # plt.savefig('./robot_path1.png')


if __name__ == '__main__':
    main()