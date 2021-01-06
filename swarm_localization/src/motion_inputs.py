#!/usr/bin/env python

'''
    This module includes all necessary classes and functions for providing motion commands to stage_ros
'''

import numpy as np
import time
import math

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Pose

import helper_functions as helper

class Motion_Inputs():
    '''
        This class includes all necessary functions for providing motion inputs to Stage
    '''
    def __init__(self, n_r, v_var, w_var, time_step):
        '''
        INPUTS:
            n_r - number of robots in swarm
            v_var - variance of 0 mean noise for linear velocity
            w_var - variance of 0 mean noise for angular velocity
        '''

        self.n_r = n_r
        self.v_var = v_var
        self.w_var = w_var
        self.time_step = time_step

        # setup publisher to all cmd_pose topics
        self.commands = []
        self.pub = [None]*self.n_r
        pub_name_temp = "cmd_pose"
        for i in range(self.n_r):
            pub_name = "robot_"+str(i)+"/"+pub_name_temp
            self.pub[i] = rospy.Publisher(pub_name, Pose, queue_size=1)
            self.commands.append(Pose2D())

        # Give some time for connection to establish
        time.sleep(1)

    def teleport_2_C(self, C_tel, robot_num):
        '''
            This function publishes all the commands to the Stage simulator
            INPUTS:
                C_tel - configuration that swarm is being teleported to
            OUTPUTS:
                None
        '''
        for command in (self.commands):
            # calculate commands based on C_tel
            command.x = C_tel[0]
            command.y = C_tel[1]
            command.theta = C_tel[2]
            # convert commands to proper format and publish
            self.pub[robot_num].publish(helper.pose2D_to_pose(command))


    def teleport_2_commands(self, C_c, v_k, w_k, robot_num):
        '''
        This function adds noise to the motion commands and teleports robots to their respective positions in the Stage simulator
        INPUTS:
        C_c - swarm's current configuration
        travel_dist_desired - desired travel distance for all robots (m), n_r by 1 matrix
        turn_angle_desired - desired turn angle for all robots (rad), n_r by 1 matrix
        OUTPUTS:
        C_achieved - the true achieved configuration of the swarm
        '''

        # add noise
        v_k_noise = v_k + np.random.normal(0, math.sqrt(self.v_var))
        actual_orientations = C_c[robot_num,2] + self.time_step * (w_k + np.random.normal(0, math.sqrt(self.w_var)))
        dx = C_c[robot_num,0] + self.time_step * math.cos(C_c[robot_num,2]) * v_k_noise
        dy = C_c[robot_num,1] + self.time_step * math.sin(C_c[robot_num,2]) * v_k_noise

        C_achieved = np.hstack((dx, dy, actual_orientations))

        # teleport swarm to positions
        self.teleport_2_C(C_achieved, robot_num)

        return C_achieved
