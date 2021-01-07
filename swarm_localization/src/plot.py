#!/usr/bin/env python

'''
    This module creates and saves plots
'''

import numpy as np
import itertools
import math
import matplotlib.pyplot as plt

def create_env_map_param(X_k_true, X_k_hat, point, time_step_num):
    '''
    This function creates plots for simulation
    INPUTS:
        X_k_true - True position of robots; datatype: np.array [num_robots, 3]
        X_k_hat - Position estimates of robots; datatype: np.array [num_robots, 3]
        point - Point-cloud data from each robot; datatype: list of np.array [num_robots][num_samples, 2]
        time_step_num - Current time-step number; datatype: int
    '''
    ox = []
    oy = []
    # plotting outer boundary
    i = 0.01
    while i < 1.81:
        ox.append(i)
        oy.append(0)
        i += 0.01
    i = 0.01
    while i < 1.81:
        ox.append(i)
        oy.append(1.20)
        i += 0.01
    i = 0
    while i < 1.21:
        ox.append(0.01)
        oy.append(i)
        i += 0.01
    i = 0
    while i < 1.21:
        ox.append(1.80)
        oy.append(i)
        i += 0.01
    i = 0
    while i < 1.21:
        ox.append(1.8)
        oy.append(i)
        i += 0.01
    # plotting inner obstacle
    i = 0.55
    while i < 1.36:
        ox.append(i)
        oy.append(0.49)
        i += 0.01
    i = 0.55
    while i < 1.36:
        ox.append(i)
        oy.append(0.78)
        i += 0.01
    i = 0.49
    while i < 0.79:
        ox.append(0.55)
        oy.append(i)
        i += 0.01
    i = 0.49
    while i < 0.79:
        ox.append(1.35)
        oy.append(i)
        i += 0.01
    
    # Specify plot proerties
    plt.figure(1, figsize=(20,10))
    plt.plot(ox, oy, ".k")
    plt.hold(True)
    
    # Plot true positions of robots
    plt.plot(X_k_true[0,0], X_k_true[0,1], marker=(3, 0, X_k_true[0,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'red')
    plt.plot(X_k_true[1,0], X_k_true[1,1], marker=(3, 0, X_k_true[1,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'red')
    plt.plot(X_k_true[2,0], X_k_true[2,1], marker=(3, 0, X_k_true[2,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'red')
    plt.plot(X_k_true[3,0], X_k_true[3,1], marker=(3, 0, X_k_true[3,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'red')

    # Plot estimate positions of robots
    plt.plot(X_k_hat[0,0], X_k_hat[0,1], marker=(3, 0, X_k_hat[0,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'blue')
    plt.plot(X_k_hat[1,0], X_k_hat[1,1], marker=(3, 0, X_k_hat[1,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'blue')
    plt.plot(X_k_hat[2,0], X_k_hat[2,1], marker=(3, 0, X_k_hat[2,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'blue')
    plt.plot(X_k_hat[3,0], X_k_hat[3,1], marker=(3, 0, X_k_hat[3,2] * (180 / math.pi) - 90), markersize=20, markerfacecolor = 'blue')
    
    # Plot point-cloud measurements for each robot
    plt.plot(point[0][:, 0], point[0][:, 1], "yo")
    plt.plot(point[1][:, 0], point[1][:, 1], "bo")
    plt.plot(point[2][:, 0], point[2][:, 1], "ro")
    plt.plot(point[3][:, 0], point[3][:, 1], "go")
    
    # Specify plot proerties
    plt.axis([-0.1, 1.91, -0.1, 1.31])
    plt.ylabel("y position")
    plt.xlabel("x position")
    plt.ion()
    plt.show()

    # Save plots at each time-step to local directory
    # plt.savefig('src/swarm_localization/results/odom_inter_outer/time_step_' + str(time_step_num) + '.png')
    
    plt.pause(0.001)
    plt.hold(False)
    plt.hold()