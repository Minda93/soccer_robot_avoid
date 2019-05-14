#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker
import pickle


def Draw_Maze(obstacles):
    """ obstacle """
    x = []
    y = []
    for pos in obstacles:
        x.append(pos[0])
        y.append(pos[1])

    ax = plt.gca()
    for i in range(len(x)):
        obstacle = patches.Rectangle((x[i]-0.225,y[i]-0.225), 0.45, 0.45, color='black')
        ax.add_patch(obstacle)

    """ goal """
    # left buttom point
    y_goal = patches.Rectangle((-3,-0.75), 0.4, 1.5, color='yellow')
    b_goal = patches.Rectangle((2.6,-0.75), 0.4, 1.5, color='blue')
    ax.add_patch(y_goal)
    ax.add_patch(b_goal)

    """ form """
    plt.grid()
    plt.ylim(-2.5, 2.5)
    plt.xlim(-3.5,3.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.show()

# =========================================================

def Define_Regions():
    regions = {}

    for i in range(1,6):
        region = {}
        region['x'] = []
        region['y'] = []
        regions[i] = region
    
    regions[1]['x'] = [-2.5,0.0]
    regions[1]['y'] = [0.225,1.775]

    regions[2]['x'] = [0.0,2.5]
    regions[2]['y'] = [0.225,1.775]

    regions[3]['x'] = [-2.5,0.0]
    regions[3]['y'] =  [-1.775,-0.225]
    
    regions[4]['x'] = [0.0,2.5]
    regions[4]['y'] = [-1.775,-0.225]

    regions[5]['x'] = [-2.5,2.5]
    regions[5]['y'] = [-0.5,0.5]

    return regions

def Random_Pose(x_range,y_range):
    
    x = np.random.uniform(x_range[0],x_range[1],1)
    y = np.random.uniform(y_range[0],y_range[1],1)
    
    return round(x[0],3),round(y[0],3)

def Check_Model_Overlapping(pos,models,error = 0.70):
    for item in models:
        if(math.hypot(pos[0]-item[0],pos[1]-item[1]) < error):
            return True
    return False

def Random_Obstacles(regions):
    obstacles = []
    flag = np.random.randint(3213)%4+1
    for i in range(1,5):
        j = 0
        while True:
            x,y = Random_Pose(regions[i]['x'],regions[i]['y'])

            if(Check_Model_Overlapping([x,y],obstacles) == False):
                obstacles.append([x,y])
                j += 1
                if(flag == i or j == 2):
                    break
    while True:
        x,y = Random_Pose(regions[i]['x'],regions[i]['y'])
        if(Check_Model_Overlapping([x,y],obstacles) == False):
            obstacles.append([x,y])
            break

    return obstacles

# =========================================================
def Read(path):
    with open(path+'env.pickle','rb') as f:
        data = pickle.load(f)
        return data

def Save(batch,path):
    with open(path+'env.pickle','wb') as f:
        pickle.dump(batch,f)

# =========================================================

def main():
    
    regions = Define_Regions()
    # obstacles = Random_Obstacles(regions)
    # Draw_Maze(obstacles)

    """ produce env """
    batch = []
    
    # seed = 9
    # np.random.seed(seed)
    # for i in range(100):
    #     batch.append(Random_Obstacles(regions))
    # Save(batch,'./{}_'.format(seed))

    """ Read env """
    seed = 9
    batch = Read('./{}_'.format(seed))
    Draw_Maze(batch[0])

if __name__ == '__main__':
    main()