#!/usr/bin/env python

'''
    This module includes all necessary classes and functions for collecting sensing data from stage_ros
'''

import numpy as np

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, Pose

import helper_functions as helper
import math

class Sensor_Data:
    '''
        This class includes all the necessary functions for obtaining and processing data from Stage
    '''

    def __init__(self, n_r, n_samples, n_samples_keep, inter_r_var, inter_b_var, outer_r_var, outer_b_var, max_sensing_range):
        '''
        INPUTS:
            n_r - number of robots in swarm
            n_samples - number of samples in the laser scan of each robot
            inter_r_var - variance for zero-mean noise that is added to the range measurement for inter-robot sensing
            inter_b_var - variance for zero-mean noise that is added to the bearing measurement for inter-robot sensing
            outer_r_var - variance for zero-mean noise that is added to the range measurement for outer-robot sensing
            outer_b_var - variance for zero-mean noise that is added to the bearing measurement for outer-robot sensing
        '''

        self.n_r = n_r
        self.n_samples = n_samples
        self.n_samples_keep = n_samples_keep
        self.inter_r_var = inter_r_var
        self.inter_b_var = inter_b_var
        self.outer_r_var = outer_r_var
        self.outer_b_var = outer_b_var
        self.max_sensing_range = max_sensing_range

        # GPS data
        self.C_k_true = np.zeros((self.n_r,3))

        # inter-robot sensing data
        self.prox_data = np.zeros((self.n_r,self.n_r))
        self.bear_data = np.zeros((self.n_r,self.n_r))

        self.prox_data_outer = np.zeros((self.n_r,self.n_r))
        self.bear_data_outer = np.zeros((self.n_r,self.n_r))
        self.prox_data_outer_v2 = np.zeros((self.n_r,self.n_r))
        self.bear_data_outer_v2 = np.zeros((self.n_r,self.n_r))

        # outer-robot measurement data, clean is data without noise
        self.outer_range = np.zeros((self.n_r,self.n_samples_keep))
        self.outer_angle = np.zeros((self.n_r,self.n_samples_keep))

        # map data
        self.map = []
        self.map_meta_data = MapMetaData()
        self.r_k = None
        self.b_k = None

    def get_data(self):
        '''
        This function subscribes to all the respective topics to get all the sensor data
        '''
        self.get_GPS_data()
        self.simulate_inter_robot_sensing()
        self.get_range_data()

    def get_GPS_data(self):
        '''
            This function subscribes to the GPS data of all robots and saves their ground truth 2D positions
            INPUTS:
                None
            OUTPUTS:
                None - self.C_k_true is updated
        '''

        sub_name_temp = "base_pose_ground_truth"
        for i in range(self.n_r):
            sub_name = "robot_"+str(i)+"/"+sub_name_temp
            data = rospy.wait_for_message(sub_name, Odometry)

            # convert to Pose2D
            pose2D_i = helper.odometry_to_pose2D(data)

            # save to C_k_true matrix
            self.C_k_true[i,0] = pose2D_i.x
            self.C_k_true[i,1] = pose2D_i.y
            self.C_k_true[i,2] = pose2D_i.theta

    def get_range_data(self):
        '''
        This function subscribes to the base_scan data of each robot and saves values to the following:
        - 'self.range_measurements'
        - 'self.angles'
        '''

        sub_name_scan_temp = "base_scan"
        for i in range(self.n_r):
            sub_name_scan = "robot_"+str(i)+"/"+sub_name_scan_temp
            data_scan = rospy.wait_for_message(sub_name_scan, LaserScan)
            range_measurements, angles = helper.range_array(data_scan)

            index = 0
            for j in range(0, self.n_samples, self.n_samples/self.n_samples_keep):
                if range_measurements[j] < self.max_sensing_range:
                    self.outer_range[i, index] = range_measurements[j] + np.random.normal(0, self.outer_r_var)
                    self.outer_angle[i, index] = angles[j] + np.random.normal(0, self.outer_b_var)
                else:
                    self.outer_range[i, index] = 0
                    self.outer_angle[i, index] = 0
                index += 1

    def get_map_data(self):
        '''
            This function subscribes to the map node and saves the map as a numpy array.
            In this array, [0,0] is the (0,0) of the map (bottom left)
        '''

        data = rospy.wait_for_message("/map", OccupancyGrid)
        self.map = helper.occupancygrid_to_numpy(data)
        self.map_meta_data = rospy.wait_for_message("map_metadata", MapMetaData)

    def simulate_inter_robot_sensing(self):
        '''
            This function simulates inter-robot sensing data, describing the relative bearing and distance between robots
            INPUTS:
                None
            OUTPUTS:
                None - self.prox_data and self.bear_data are updated
        '''

        # get GPS data from stage
        self.get_GPS_data()

        x_data = self.C_k_true[:,0].reshape(-1,1)
        y_data = self.C_k_true[:,1].reshape(-1,1)
        theta_data = self.C_k_true[:,2].reshape(-1,1)

        # calculate inter-robot distance in cartesian coordinations
        dxs = x_data.T - x_data
        dys = y_data.T - y_data

        # transform to polar coordinations
        prox_data_temp, bear_data_temp = helper.cart2pol(dxs,dys)

        # add noise
        prox_data = prox_data_temp + np.random.normal(0, (10*self.inter_r_var/3)/1000, (self.n_r, self.n_r))
        bear_data = bear_data_temp - theta_data + np.random.normal(0, (10*self.inter_b_var/3)*np.pi/180, (self.n_r, self.n_r))

        # get rid of unnecessary data
        prox_data = prox_data - np.diag(np.diag(prox_data))
        bear_data = bear_data - np.diag(np.diag(bear_data))

        self.prox_data = prox_data
        self.bear_data = bear_data
    
    def simulate_inter_robot_sensing_outer(self, X_k_hat_outer):
        '''
            This function simulates inter-robot sensing data, describing the relative bearing and distance between robots
            INPUTS:
                None
            OUTPUTS:
                None - self.prox_data and self.bear_data are updated
        '''

        x_data = X_k_hat_outer[:,0].reshape(-1,1)
        y_data = X_k_hat_outer[:,1].reshape(-1,1)
        theta_data = X_k_hat_outer[:,2].reshape(-1,1)

        # calculate inter-robot distance in cartesian coordinations
        dxs = x_data.T - x_data
        dys = y_data.T - y_data

        # transform to polar coordinations
        prox_data_temp, bear_data_temp = helper.cart2pol(dxs,dys)

        # add noise
        prox_data = prox_data_temp + np.random.normal(0, (10*self.inter_r_var/3)/1000, (self.n_r, self.n_r))
        bear_data = bear_data_temp - theta_data + np.random.normal(0, (10*self.inter_b_var/3)*np.pi/180, (self.n_r, self.n_r))

        # get rid of unnecessary data
        prox_data = prox_data - np.diag(np.diag(prox_data))
        bear_data = bear_data - np.diag(np.diag(bear_data))

        self.prox_data_outer = prox_data
        self.bear_data_outer = bear_data

    def simulate_inter_robot_sensing_outer_v2(self, X_k_hat, X_k_true):
        '''
            This function simulates inter-robot sensing data, describing the relative bearing and distance between robots
            INPUTS:
                None
            OUTPUTS:
                None - self.prox_data and self.bear_data are updated
        '''

        x_data_hat = X_k_hat[:,0].reshape(-1,1)
        y_data_hat = X_k_hat[:,1].reshape(-1,1)
        theta_data_hat = X_k_hat[:,2].reshape(-1,1)

        x_data_true = X_k_true[:,0].reshape(-1,1)
        y_data_true = X_k_true[:,1].reshape(-1,1)
        # theta_data_true = X_k_true[:,2].reshape(-1,1)

        # calculate inter-robot distance in cartesian coordinations
        # dxs = x_data_hat.T - x_data_true
        # dys = y_data_hat.T - y_data_true
        dxs = x_data_hat.T - x_data_true
        dys = y_data_hat.T - y_data_true

        # transform to polar coordinations
        prox_data_temp, bear_data_temp = helper.cart2pol(dxs,dys)
        for i in range(np.shape(X_k_hat)[0]):
            for j in range(np.shape(X_k_hat)[0]):
                bear_data_temp[j, i] = helper.wraptopi(bear_data_temp[j, i])

        # add noise
        prox_data = prox_data_temp + np.random.normal(0, (10*self.inter_r_var/3)/1000, (self.n_r, self.n_r))
        bear_data = bear_data_temp - theta_data_hat + np.random.normal(0, (10*self.inter_b_var/3)*np.pi/180, (self.n_r, self.n_r))

        # get rid of unnecessary data
        prox_data = prox_data - np.diag(np.diag(prox_data))
        bear_data = bear_data - np.diag(np.diag(bear_data))

        self.prox_data_outer_v2 = prox_data
        self.bear_data_outer_v2 = bear_data

    def simulate_inter_robot_sensing_outer_v3(self, X_k_hat, X_k_true, robot_num):
        '''
            This function simulates inter-robot sensing data, describing the relative bearing and distance between robots
            INPUTS:
                None
            OUTPUTS:
                None - self.prox_data and self.bear_data are updated
        '''
        r_k = np.zeros((np.shape(X_k_true)[0], 1))
        b_k = np.zeros((np.shape(X_k_true)[0], 1))
        for i in range(np.shape(X_k_true)[0]):
            dx = X_k_true[i, 0] - X_k_hat[robot_num, 0]
            dy = X_k_true[i, 1] - X_k_hat[robot_num, 1]
            r_k[i, 0] = math.sqrt(pow(dx, 2) + pow(dy, 2)) + np.random.normal(0, self.inter_r_var)
            b_k[i, 0] = math.atan2(dy, dx) - X_k_hat[robot_num, 2] + np.random.normal(0, self.inter_b_var)
            b_k[i, 0] = helper.wraptopi(b_k[i, 0])
        self.r_k = r_k
        self.b_k = b_k