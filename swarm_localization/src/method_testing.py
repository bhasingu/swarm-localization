#!/usr/bin/env python

'''
    This module tests the methodology
'''

#------------------------------------------
# ros related imports
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Pose
#------------------------------------------
# Other library imports
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import math
#------------------------------------------
# Import created scripts
import helper_functions as helper
from sensor_data import Sensor_Data
import localization as localization
import landmark_detection as landmark_detection
from motion_inputs import Motion_Inputs
import motion_commands as motion_commands
import plot as plot
#------------------------------------------

def method_testing():

    # initialize node
    rospy.init_node('AER1515_method_testing', anonymous = True)

    # create swarm
    n_r = 4
    n_samples = 1000
    # n_samples_keep is the number of samples we want to retain from the n_samples.
    # The reason for doing this is due to the bias in the range measurements presented in stage_ros when the number of samples per scan is smaller than 500
    n_samples_keep = 20

    # Noise values
    # ----------------------------------
    # inter_r_var = 0.000900360036
    # inter_b_var = 0.000671431744
    # outer_r_var = 0.000900360036
    # outer_b_var = 0.000671431744
    # v_var = 0.004420255225
    # w_var = 0.008186087529
    # ----------------------------------
    # inter_r_var = pow(0.005, 2)
    # inter_b_var = pow(5*math.pi/180, 2)
    # outer_r_var = pow(0.005, 2)
    # outer_b_var = pow(5*math.pi/180, 2)
    # v_var = pow(0.005, 2)
    # w_var = 0.008186087529
    # ----------------------------------
    # inter_r_var = 0
    # inter_b_var = 0
    # outer_r_var = 0
    # outer_b_var = 0
    # v_var = 0
    # w_var = 0
    # ----------------------------------
    inter_r_var = pow(0.001, 2)
    inter_b_var = pow(1*math.pi/180, 2)
    outer_r_var = pow(0.001, 2)
    outer_b_var = pow(1*math.pi/180, 2)
    v_var = pow(0.001, 2)
    w_var = 0.008186087529

    time_step = 0.1 # Frequency of get information and updates in seconds
    max_sensing_range = 0.5 # Maximum sensing range of laser scan

    # Linear and angular velocity inputs for robots in swarm
    robot_vel, num_time_step = motion_commands.create_motion_commands()

    # Position coordinates for known corner landmarks in the environment
    landmark_pos_outer = [[0.55, 0.49],
                    [0.55, 0.78],
                    [1.35, 0.78],
                    [1.35, 0.49],
                    [1.80, 0.0],
                    [1.80, 1.20],
                    [0.01, 1.20],
                    [0.01, 0.0]]
    landmark_pos_array = np.asarray(landmark_pos_outer).reshape((len(landmark_pos_outer), 2))
    # Get sensor data from robots to get intial sensor readings
    sensor_data = Sensor_Data(n_r, n_samples, n_samples_keep, inter_r_var, inter_b_var, outer_r_var, outer_b_var, max_sensing_range)
    sensor_data.get_data()

    print("Ground truth pose")
    print(sensor_data.C_k_true)

    # Initialize initial pose of robots and their covariance belief matrix
    X_k_true = copy.deepcopy(sensor_data.C_k_true)
    X_k_check = np.zeros((n_r,3))
    P_k_check = np.zeros((3*n_r, 3))
    X_k_hat_outer = np.zeros((n_r,3))
    P_k_hat_outer = np.zeros((3*n_r, 3))
    X_k_hat_inter = np.zeros((n_r,3))
    X_k_1_hat = X_k_true  # Assuming inital state known for robots
    P_k_1_hat = np.zeros((3*n_r, 3))
    P_k_1_hat[0,0] = 1
    P_k_1_hat[1,1] = 1
    P_k_1_hat[2,2] = 0.1
    P_k_1_hat[3,0] = 1
    P_k_1_hat[4,1] = 1
    P_k_1_hat[5,2] = 0.1
    P_k_1_hat[6,0] = 1
    P_k_1_hat[7,1] = 1
    P_k_1_hat[8,2] = 0.1

    error_sum = np.zeros((n_r,3))

    for t in range(num_time_step):
        P = np.zeros((n_r-1, 2))
        Q = np.zeros((n_r-1, 2))
        p = np.zeros((n_r-1, 2))
        q = np.zeros((n_r-1, 2))
        for robot_num in range(n_r):
            valid_index_inter = []
            for j in range(n_r):
                if j != robot_num:
                    valid_index_inter.append(j)

            sensor_data.simulate_inter_robot_sensing(X_k_1_hat, X_k_true, robot_num)
            x, y =helper.pol2cart(copy.deepcopy(sensor_data.prox_data), copy.deepcopy(sensor_data.bear_data))

            for k, index in enumerate(valid_index_inter):
                q[k, 0] = x[index][0]
                q[k, 1] = y[index][0]
            Q = helper.rotate(q, X_k_1_hat[robot_num, 2])
            Q[:, 0] += X_k_1_hat[robot_num, 0]
            Q[:, 1] += X_k_1_hat[robot_num, 1]
            v_k = robot_vel[t][robot_num,0]
            w_k = robot_vel[t][robot_num,1]
            T = time_step
            motion = Motion_Inputs(n_r, v_var, w_var, T)
            motion.teleport_2_commands(X_k_true, v_k, w_k, robot_num)
            ekf_i = localization.EKF(T, v_var, w_var, inter_r_var, inter_b_var)
            X_k_check[robot_num,:], P_k_check[3*robot_num:3*robot_num+3,:] = ekf_i.predict(X_k_1_hat[robot_num,:].reshape((3,1)), P_k_1_hat[3*robot_num:3*robot_num+3,:], v_k, w_k)
           
            sensor_data.get_data()
            sensor_data.simulate_inter_robot_sensing(X_k_check, X_k_true, robot_num)
            x, y = helper.pol2cart(copy.deepcopy(sensor_data.prox_data), copy.deepcopy(sensor_data.bear_data))
            for k, index in enumerate(valid_index_inter):
                p[k, 0] = x[index][0]
                p[k, 1] = y[index][0]
            P = helper.rotate(p, X_k_check[robot_num, 2])
            P[:, 0] += X_k_check[robot_num, 0]
            P[:, 1] += X_k_check[robot_num, 1]
            
            rot_angle, trans_x, trans_y = landmark_detection.point_based_matching(P, Q)
            
            X_k_hat_inter[robot_num, 0:2] = helper.rotate(X_k_check[robot_num, 0:2].reshape((1, 2)), rot_angle)
            X_k_hat_inter[robot_num, 0] += trans_x
            X_k_hat_inter[robot_num, 1] += trans_y
            X_k_hat_inter[robot_num, 2] = X_k_check[robot_num, 2] + rot_angle

            x, y = helper.pol2cart(copy.deepcopy(sensor_data.outer_range), copy.deepcopy(sensor_data.outer_angle))
            point = landmark_detection.convert2world(x,y,X_k_hat_inter)
            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(point[robot_num])
            group = landmark_detection.groupCluster(pcd_i)
            print("grouped clusters")
            print(group)
            
            intersects = np.array([])
            for j in group:
                line = landmark_detection.HoughLines(j)
                print("lines")
                print(line)
                intersect = landmark_detection.findIntersect(line)
                intersects = landmark_detection.concat_potential_lm(intersect, intersects)
            print("Intersects")
            print(intersects)
            closest_pt2lm = landmark_detection.checkNearestLandmark(intersects, landmark_pos_outer)
            print("closest_pt2lm")
            print(closest_pt2lm)
            outer_meas, valid_index_outer = landmark_detection.robot2seen_lm(X_k_hat_inter[robot_num, :], closest_pt2lm)
            r_k_outer = outer_meas[:,0].reshape((np.shape(landmark_pos_array)[0],1))
            b_k_outer = outer_meas[:,1].reshape((np.shape(landmark_pos_array)[0],1))
            if v_k == 0.0 and w_k == 0.0:
                print("BOTH ARE ZERO")
                X_k_hat_outer[robot_num, :] = X_k_hat_inter[robot_num,:]
                P_k_hat_outer[3*robot_num:3*robot_num+3,:] = P_k_check[3*robot_num:3*robot_num+3,:]
            else:
                X_k_hat_outer[robot_num, :], P_k_hat_outer[3*robot_num:3*robot_num+3,:] = ekf_i.update(X_k_hat_inter[robot_num, :].reshape((3,1)), P_k_check[3*robot_num:3*robot_num+3,:], landmark_pos_array, valid_index_outer, r_k_outer, b_k_outer)

            error_sum[robot_num,0] += pow((X_k_hat_outer[robot_num,0] - X_k_true[robot_num,0]), 2)
            error_sum[robot_num,1] += pow((X_k_hat_outer[robot_num,1] - X_k_true[robot_num,1]), 2)
            error_sum[robot_num,2] += pow((helper.wraptopi(X_k_hat_outer[robot_num,2] - X_k_true[robot_num,2])), 2)
        

        # Loop start: robot move, predict pose with motion model, update/correct with mesurement model
        sensor_data.get_data()
        X_k_true = copy.deepcopy(sensor_data.C_k_true)

        print("X_k_true")
        print(X_k_true)
        print("X_k_hat_outer")
        print(X_k_hat_outer)

        # Plot the results
        plot.create_env_map_param(X_k_true, X_k_hat_outer, point, t)
        
        # Update the pose belief values
        X_k_1_hat = X_k_hat_outer.reshape((n_r,3))
        P_k_1_hat = P_k_hat_outer.reshape((3*n_r,3))
    
    for i in range(n_r):
        error_sum[i,0] = math.sqrt(error_sum[i,0]/num_time_step)
        error_sum[i,1] = math.sqrt(error_sum[i,1]/num_time_step)
        error_sum[i,2] = math.sqrt(error_sum[i,2]/num_time_step)
    print("RMSE Error")
    print(error_sum)


if __name__ == '__main__':
    method_testing()