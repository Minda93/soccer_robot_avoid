#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

import math
import numpy as np
import matplotlib.pyplot as plt

r""" 
    log
        episode
        reward 
        total step
        info 
"""

def Read_File(directory,fileName = 'log.txt'):
    path = directory + fileName
    with open(path,'r') as f:
        data = []
        for line in f.readlines():
            """  """
            line = line.strip()
            
            """ remove special char """
            line = line.replace("[","")
            line = line.replace("]","")

            line = line.splitlines()
            line = line[0].split(",")
            data.append(line)
        return data
    return None
def Plot_Reward(data,choice = "reward"):
    if(data is not None):

        if(choice == "reward"):
            """ reward """
            reward = []
            for item in data:
                reward.append([float(item[1])])

            # mean = np.mean(reward,axis=1)
            # std =  np.std(reward)
            # plt.fill_between(np.arange(1,len(reward_)+1, 1), mean - std,
            #              mean + std, alpha=0.1, color="g")
            # plt.plot(np.arange(1,len(reward_)+1, 1)*10,mean)

            reward_ = []
            i = 1
            ep_r = 0
            for r in reward:
                ep_r += r[0]
                if(i%10 == 0):
                    reward_.append([ep_r/10.])
                    ep_r = 0
                i += 1
            mean = np.mean(reward_,axis=1)
            std =  np.std(reward_)
            plt.plot(np.arange(1,len(reward_)+1, 1)*10,reward_)
            # plt.plot(np.arange(1,len(reward)+1, 1),reward)
            plt.ylabel('reward', fontsize=18)
        elif(choice == "loss"):

            """ loss """
            loss = []
            for item in data:
                if(float(item[2]) != 0):
                    loss.append([float(item[len(item)-1])/float(item[2])])
                
            loss_ = []
            i = 1
            ep_l = 0
            for l in loss:
                ep_l += l[0]
                if(i%10 == 0):
                    loss_.append([ep_l/10.])
                    ep_l = 0
                i += 1
            plt.plot(np.arange(1,len(loss_)+1, 1)*10,loss_,dashes=[6, 2])
            # plt.plot(np.arange(1,len(loss)+1, 1),loss)
            plt.ylabel('loss', fontsize=18)

        plt.xlabel('episode', fontsize=18)
        plt.tight_layout()
        plt.show()
        # plt.savefig('./reward_test.png')

def main():
    # data_1 = Read_File('./','log_1.txt')
    # data_2 = Read_File('./','log_2.txt')
    # data_3 = Read_File('./','log_3.txt')
    # data_4 = Read_File('./','log_4.txt')
    # data = np.concatenate((np.array(data_1),np.array(data_2)))
    # data = np.concatenate((np.array(data),np.array(data_3)))
    # data = np.concatenate((np.array(data),np.array(data_4)))
    
    data = Read_File('./','log.txt')
    
    # Plot_Reward(data)
    Plot_Reward(data,"loss")


if __name__ == '__main__':
    main()
