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
            n_r - Number of robots in swarm; datatype: int
            n_samples - Number of samples in the laser scan of each robot from Stage_ROS; datatype: int
            n_samples_keep - Number of samples we want to retain; datatype: int
            inter_r_var - Variance for zero-mean noise that is added to the range measurement for inter-robot sensing; datatype: float
            inter_b_var - Variance for zero-mean noise that is added to the bearing measurement for inter-robot sensing; datatype: float
            outer_r_var - Variance for zero-mean noise that is added to the range measurement for outer-robot sensing; datatype: float
            outer_b_var - Variance for zero-mean noise that is added to the bearing measurement for outer-robot sensing; datatype: float
            max_sensing_range - Maximum sensing range of the laser scan; datatype: float
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
        # self.prox_data = np.zeros((self.n_r,self.n_r))
        # self.bear_data = np.zeros((self.n_r,self.n_r))
        self.prox_data = None
        self.bear_data = None

        # outer-robot measurement data, clean is data without noise
        self.outer_range = np.zeros((self.n_r,self.n_samples_keep))
        self.outer_angle = np.zeros((self.n_r,self.n_samples_keep))

        # map data
        self.map = []
        self.map_meta_data = MapMetaData()

    def get_data(self):
        '''
        This function subscribes to all the respective topics to get all the sensor data
        '''
        self.get_GPS_data()
        # self.simulate_inter_robot_sensing()
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

    def simulate_inter_robot_sensing(self, X_k_hat, X_k_true, robot_num):
        '''
            This function simulates inter-robot sensing data, describing the relative bearing and distance between robots
            INPUTS:
                X_k_hat - Pose estimates for all robots current time-step; datatype: np.array[number_robots = 4, 3]
                X_k_true - True pose for all robots current time-step; datatype: np.array[number_robots = 4, 3]
                robot_num - robot of interest; datatype: int
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
        self.prox_data = r_k
        self.bear_data = b_k