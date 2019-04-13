#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

def Get_X_Euler(x,y,z,w,state='zyx'):
    if(state == 'zyx'):
        return round(math.atan2(2 * (y*z + w*x), w*w - x*x - y*y + z*z)/math.pi*180,2)

def Get_Y_Euler(x,y,z,w,state='zyx'):
    if(state == 'zyx'):
        return round(math.asin(-2 * (x*z - w*y))/math.pi*180,2)

def Get_Z_Euler(x,y,z,w,state='zyx'):
    if(state == 'zyx'):
        return round(math.atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z)/math.pi*180,2)

def Get_Orientation(x,y,z,state='zyx'):
    if(state == 'zyx'):
        x_ = math.radians(x/2.0)
        y_ = math.radians(y/2.0)
        z_ = math.radians(z/2.0)
        
        qx = math.sin(x_)*math.cos(y_)*math.cos(z_)-math.cos(x_)*math.sin(y_)*math.sin(z_)
        qy = math.cos(x_)*math.sin(y_)*math.cos(z_)+math.sin(x_)*math.cos(y_)*math.sin(z_)
        qz = math.cos(x_)*math.cos(y_)*math.sin(z_)-math.sin(x_)*math.sin(y_)*math.cos(z_)
        qw = math.cos(x_)*math.cos(y_)*math.cos(z_)+math.sin(x_)*math.sin(y_)*math.sin(z_)
        return qx,qy,qz,qw