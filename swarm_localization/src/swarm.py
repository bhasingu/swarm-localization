#!/usr/bin/env python

'''
    This module includes all necessary classes and functions for creating and controlling a swarm
'''

import numpy as np
import time

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Pose, Twist

import helper_functions as helper
from motion_inputs import Motion_Inputs
import math

class Robot:
    
    def __init__(self, n_r, id, x, y, theta, v_var, w_var, inter_r_var, inter_b_var, outer_r_var, outer_b_var): # x, y, theta is the true pose of the robot at time t.
        self.id = id

        # self.motion_noise = 0.01
        # self.measurement_noise = 0.005
        self.v_var = v_var
        self.w_var = w_var 
        self.inter_r_var = inter_r_var 
        self.inter_b_var = inter_b_var 
        self.outer_r_var = outer_r_var 
        self.outer_b_var = outer_b_var
        # ???
        self.pose_t_init = np.array([x,y,theta]) # robot's initial true configuration (at t=0)
        # print(self.pose_t_init)
        self.pose_t_init_hat = np.zeros((1,3)) # robot's initial estimated true configuration (t=0)

        # might change following to vector.twist class
        self.pose_t = np.zeros((1,3)) # Matrix of true position of robot in swarm at a certain time

        self.pose_t_hat = np.zeros((1,3)) # Matrix of final position estimate of robots in swarm at time 't'. In other words, this is the Posterior estimate
        self.pose_t_u_hat = np.zeros((1,3)) # Matrix of estimated position of robot using odometry measurement at time t
        self.top_t_hat = np.zeros((n_r,3)) # robot's estimated inter-robot sensing wrt other robots

        # motion variable for CTRV model
        self.twist = Twist()

        self.pub = rospy.Publisher("robot_"+str(id)+"/cmd_vel", Twist, queue_size=1)

        #self.ukf
        #self.ekf
        #self.pf

    def move(self):
        noisy_twist = Twist()
        noisy_twist.linear.x = self.twist.linear.x + self.noise(self.v_var)
        noisy_twist.angular.z = self.twist.angular.z + self.noise(self.w_var)
        self.pub.publish(noisy_twist)

    def noise(self, var):
        return np.random.normal(0, math.sqrt(var), 1)[0] # noise for motion model, assume 0 mean


class Swarm:
    def __init__(self, n_r, C_t_init, v_var, w_var, inter_r_var, inter_b_var, outer_r_var, outer_b_var, time_step): # Every time you declare a variable using this class, have to mention number of robots in swarm
        self.n_r = n_r # This is the number of robots present in the swarm
        self.timestep = time_step # default timestep update
        self.Robot = []

        self.C_t = np.zeros((n_r,3)) # Matrix of true positions of all robots in swarm at a certain time
        self.C_t_hat = np.zeros((n_r,3)) # Matrix of final position estimate of all robots in swarmat time 't'. In other words, this is the Prior estimate
        self.C_t_u_hat = np.zeros((n_r,3)) # Matrix of estimated position of all robots using odometry measurements at time t
        self.T_t_p_hat = np.zeros((n_r,3)) # swarm's estimated topology based on inter-robot measurements at time t

        self.C_t_init = C_t_init # swarm's initial true configuration (at t=0)

        # process and measurement noise variance
        self.v_var = v_var
        self.w_var = w_var 
        self.inter_r_var = inter_r_var 
        self.inter_b_var = inter_b_var 
        self.outer_r_var = outer_r_var 
        self.outer_b_var = outer_b_var
        # self.C_t_init_hat = np.zeros((n_r,3)) # swarm's initial estimated true configuration (t=0)

    def update_robot_pose(self, C_t):
        for i in range(self.n_r):
            R = Robot(self.n_r, i, C_t[i, 0], C_t[i, 1], C_t[i, 2], self.v_var, self.w_var, self.inter_r_var, self.inter_b_var, self.outer_r_var, self.outer_b_var)
            self.Robot.append(R)
            
    def move(self, timestep = None):
        if timestep == None:
            timestep = self.timestep
        beginTime = rospy.Time.now()
        duration = rospy.Duration(timestep)
        endTime = beginTime + duration
        print("Begin Swarm Moving")
        while rospy.Time.now() < endTime:
            for r in self.Robot:
                r.move()
        print("Finish Swarm Moving")

    def init_random_configuration(self, min_dist):
        '''
            This function randomly initializes the configuration of the swarm
            INPUTS:
                min_dist - minimum distance between robots for initialization
            OUTPUTS:
                C_t_init - swarm's initial true configuration
        '''
        C_t_init = np.zeros((1,3))
        for i in range(1,self.n_r):
            d = np.random.uniform(min_dist, 2*min_dist)
            theta = np.random.uniform(-np.pi, np.pi)
            dx, dy = helper.pol2cart(d,theta)

            n_created = C_t_init.shape[0]
            rand_robot_id = np.random.randint(0,n_created)
            pose_temp = C_t_init[rand_robot_id,:] + np.array([dx, dy, theta])

            C_t_init = np.vstack((C_t_init, pose_temp))

            self.C_t.append(C_t_init)
            self.C_t_init = C_t_init
        print("C_t_init shape: ")
        return C_t_init

    def init_random_configuration_map(self, min_dist, map, map_res):
        '''
            This function randomly initializes the configuration of the swarm within the boundary of the given map, while also considering obstacles
            INPUTS:
                min_dist - minimum distance between robots for initialization
                map - occupancy grid map of the world, __ by __ np array
                map_res - resolution of the map (m/pixel)
            OUTPUTS:
                C_t_init - swarm's initial true configuration
        '''

        # randomly initialize:
        C_t_init = self.init_random_configuration(min_dist)

        # move swarm within free map region
        min_x = 0
        min_y = 0
        max_x = map.shape[1]*map_res
        max_y = map.shape[0]*map_res

        found = False
        while found is False:
            # randomly translate swarm
            x_temp = np.random.uniform(min_x, max_x)
            y_temp = np.random.uniform(min_y, max_y)
            C_t_init_temp = C_t_init[:,0:2] + np.array([x_temp, y_temp])

            # check if within map bounds
            found = helper.check_within_map(C_t_init_temp[:,0], C_t_init_temp[:,1], map, map_res)

        C_t_init[:,0:2] = C_t_init_temp

        return C_t_init