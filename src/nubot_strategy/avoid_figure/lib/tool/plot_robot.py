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
    robots = Read_Data("./log_robot_show.pickle")
    paths = Read_Data("log_path_show.pickle")
    obstalces = Read_Data("log_obstacle_show.pickle")
    choise_episode = 1
    path = paths[choise_episode]
    obstalce = obstalces[choise_episode]

    """ add plot object """
    ax = plt.gca()

    """ path """
    x = []
    y = []
    for pos in path:
        x.append(pos[0])
        y.append(pos[1])
    plt.plot(x,y,color='red')

    """ obstacle """
    x = []
    y = []
    for pos in obstalce:
        x.append(pos[0])
        y.append(pos[1])
    print(x,y)
    
    for i in range(len(x)):
        obstacle = patches.Rectangle((x[i],y[i]), 0.4, 0.4, color='black')
        ax.add_patch(obstacle)

    """ goal """
    # left buttom point
    y_goal = patches.Rectangle((-3,-0.75), 0.4, 1.5, color='yellow')
    b_goal = patches.Rectangle((2.6,-0.75), 0.4, 1.5, color='blue')
    ax.add_patch(y_goal)
    ax.add_patch(b_goal)

    plt.ylim(-2.5, 2.5)
    plt.xlim(-3.5,3.5)
    plt.show()
    # plt.savefig('./robot_path1.png')


if __name__ == '__main__':
    main()