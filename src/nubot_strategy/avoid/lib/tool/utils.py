#!/usr/bin/env python3
# -*- coding: utf-8 -*-+

import numpy as np
from collections import deque
import random

class ReplayBuffer_Deque(object):
    r"""
        (state, action, reward, next_state, done)  
    """
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def Sample(self,batch_size):
        batch = []
        if(self.num_experiences < batch_size):
            batch = random.sample(self.buffer, self.num_experiences)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        s_batch = [e[0] for e in batch]
        a_batch = np.array([e[1] for e in batch])
        r_batch = np.array([e[2] for e in batch])
        s2_batch = [e[3] for e in batch]
        d_batch = np.array([e[4] for e in batch])

        return s_batch,a_batch,r_batch,s2_batch,d_batch
    
    def Action_Sample(self,step):
        if(self.num_experiences):
            return np.array([self.buffer[step][1]])
    
    def Get_Buffer_Size(self):
        return self.buffer_size
    
    def Add(self,state,action,reward,new_state,done):
        experience = (state,action,reward,new_state,done)
        if(self.num_experiences < self.buffer_size):
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    
    def Get_Count(self):
        return self.num_experiences
    
    def Clear(self):
        self.buffer.clear()
        self.num_experiences = 0

# ==============================================================================

class ReplayBuffer(object):
    r"""
        (state, action, reward, next_state, done)  
    """
    def __init__(self,buffer_size = 1e6):
        self.storage = []
        self.buffer_size = buffer_size
        self.ptr = 0

    def Add(self,data):
        if(len(self.storage) == self.buffer_size):
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.buffer_size
        else:
            self.storage.append(data)

    def Sample(self,batch_size):
        ind = np.random.randint(0, len(self.storage), size = batch_size)
        s_t, a, r, s_t1, d = [], [], [], [], []

        for i in ind:
            S_t, A, R, S_t1, D = self.storage[i]

            s_t.append(np.array(S_t, copy = False))
            a.append(np.array(A, copy = False))
            r.append(np.array(R, copy = False))
            s_t1.append(np.array(S_t1, copy = False))
            d.append(np.array(D, copy = False))

        return np.array(s_t), np.array(a), np.array(r), np.array(s_t1),np.array(d)