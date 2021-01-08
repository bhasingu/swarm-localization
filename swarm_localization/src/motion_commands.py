#!/usr/bin/env python

'''
    This module creates the motion commands list
'''

import numpy as np
import math

def create_motion_commands():
    '''
    This creates a list for linear and angular motion commands for each of the 4 robots in the swarm
    INPUTS:
        NONE
    OUTPUTS:
        robot_vel - list of commands; datatype: list of np.array [num_robots = 4, 2]
    '''
    robot_vel = []
    for i in range(20):
        robot_vel.append(np.array([[0.05, 0],
                [0.05, 0],
                [0.05, 0],
                [0.05, 0]]))

    for i in range(20):
        robot_vel.append(np.array([[0, math.pi/4],
                [0.05, 0],
                [0.05, 0],
                [0, math.pi/4]]))

    for i in range(10):
        robot_vel.append(np.array([[0, math.pi/4],
                [0, -math.pi/4],
                [0, -math.pi/4],
                [0, math.pi/4]]))

    for i in range(10):
        robot_vel.append(np.array([[0, -math.pi/4],
                [0, math.pi/4],
                [0, math.pi/4],
                [0, -math.pi/4]]))
    
    for i in range(20):
        robot_vel.append(np.array([[0, -math.pi/4],
                [0.05, 0],
                [0.05, 0],
                [0, -math.pi/4]]))

    for i in range(51):
        robot_vel.append(np.array([[0.05, 0],
                [0.05, 0],
                [0.05, 0],
                [0.05, 0]]))

    for i in range(10):
        robot_vel.append(np.array([[0.05, 0],
                [0, 0],
                [0, 0],
                [0.05, 0]]))

    for i in range(30):
        robot_vel.append(np.array([[0.05, 0],
                [0.05, 0],
                [0.05, 0],
                [0.05, 0]]))

    for i in range(10):
        robot_vel.append(np.array([[0.05, 0],
                [0.05, 0],
                [0, -math.pi/4],
                [0.05, 0]]))

    for i in range(30):
        robot_vel.append(np.array([[0.05, 0],
                [0, -math.pi/4],
                [0, -math.pi/4],
                [0.05, 0]]))
    
    for i in range(10):
        robot_vel.append(np.array([[0.05, 0],
                [0, math.pi/4],
                [0, 0],
                [0.05, 0]]))

    for i in range(20):
        robot_vel.append(np.array([[0.05, 0],
                [0, 0],
                [0, 0],
                [0.05, 0]]))
    
    for i in range(40):
        robot_vel.append(np.array([[0, math.pi/4],
                [0, 0],
                [0, 0],
                [0, math.pi/4]]))
    
    return robot_vel, len(robot_vel)