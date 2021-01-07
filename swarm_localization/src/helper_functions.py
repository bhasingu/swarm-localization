#!/usr/bin/env python

'''
    This module includes some useful helper functions and imports the necessary libraries
'''

#------------------------------------------
# ros related imports
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose2D, Pose, Quaternion
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
#------------------------------------------
# Other library imports
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import math
#------------------------------------------

def wraptopi(theta):
    '''
    This function bounds angles between [0, pi] and [-pi, 0]
    '''
    return (theta + np.pi) % (2*np.pi) - np.pi
    
def range_array(data_scan):
    '''
    This function extracts the range and bearing measurement values from ROS LaserScan message 
    INPUTS:
        data_scan - Laserscan; datatype: ROS type sensor_msgs/LaserScan
    OUTPUTS:
        ranges - Range measurement to environment; datatype: np.array [num_samples, 1]
        angles - Bearing measurement to environment; datatype: np.array [num_samples, 1]
    '''
    # The range measurements array for a robot
    ranges = np.asarray(data_scan.ranges, dtype=np.float32)

    # The angle wrt to the robot's x axis of the first range measurement element in array
    angle_min = data_scan.angle_min

    # The angle increments
    angle_increment = data_scan.angle_increment

    # The number of samples in the range array for a robot
    n_samples = len(ranges)

    # The angle associated with each sample in the range measurement array
    angles = np.zeros(n_samples)

    for i in range(n_samples):
        angles[i] = angle_min + angle_increment*i

    return ranges, angles

def odometry_to_pose2D(pose3D):
    '''
        This function transforms the 3D pose of ROS type nav_msgs.Odometry to 2D pose of ROS type geometry_msgs.Pose2D (only applicable when operating in 2D)
        INPUTS:
            pose3D - 3D pose of robot; datatype: ROS type nav_msgs.Odometry
        OUTPUTS:
            pose2D - 2D pose of robot; datatype: ROS type geometry_msgs.Pose2D
    '''
    rot_data = pose3D.pose.pose.orientation
    pos_data = pose3D.pose.pose.position

    euler_angles = euler_from_quaternion(quaternion=(rot_data.x, rot_data.y, rot_data.z, rot_data.w))

    pose2D = Pose2D()
    pose2D.theta = euler_angles[2]
    pose2D.x = pos_data.x
    pose2D.y = pos_data.y

    return pose2D

def pose2D_to_pose(pose2D):
    '''
        This function transforms the 2D pose of ROS type geometry_msgs.Pose2D to 3D pose of ROS type geometry_msgs.Pose2D
        INPUTS:
            pose2D - 2D pose of robot; datatype: ROS type geometry_msgs.Pose2D
        OUTPUTS:
            pose - 3D pose of robot; datatype: ROS type geometry_msgs.Pose
    '''
    pose = Pose()

    pose.position.x = pose2D.x
    pose.position.y = pose2D.y
    pose.position.z = 0

    q = quaternion_from_euler(0.0, 0.0, pose2D.theta)
    pose.orientation = Quaternion(*q)

    return pose

def cart2pol(x,y):
    '''
        This function converts from cartesian coordinations to polar coordinations
        INPUT: cartesian coordinates
            x - n by m matrix of all x coordinates
            y - n by m matrix of all y coordinates
        OUTPUT: polar coordinates
            d - n by m matrix of all distance values
            theta - n by m matrix of all angle values, ranging from (-pi, pi)
    '''
    d = np.sqrt(np.square(x) + np.square(y))
    theta = np.arctan2(y,x)

    return d, theta

def pol2cart(r, theta):
    '''
        This function converts coordinates from polar to cartesian representation
        INPUT: polar coordinates
            r - n by m matrix of all distance values
            theta - n by m matrix of all angle values, ranging from (-pi, pi)
        OUTPUT: cartesian coordinates
            x_list - n by m matrix of all x coordinates
            y_list - n by m matrix of all y coordinates
    '''
    x_list = []
    y_list = []
    n_r = np.shape(r)[0]
    n_samples = np.shape(r)[1]
    for i in range(n_r):
        x_i_list = []
        y_i_list = []
        for j in range(n_samples):
            if r[i,j] > 0.001:
                x_i_list.append(r[i,j] * math.cos(theta[i,j]))
                y_i_list.append(r[i,j] * math.sin(theta[i,j]))
        x_list.append(x_i_list)
        y_list.append(y_i_list) 

    return x_list,y_list

def rotate(position_L, theta_LG):
    '''
        This is the rotation function in 2D space
        INPUTS:
            position_L - _ by 2 np array of positions [x, y], defined with respect to frame L
            theta_LG - rotation angle, defining the angle from frame G to frame L
        OUTPUTS:
            position_G - rotated positions with respect to frame G, _ by 2 np array
    '''
    c, s = np.cos(theta_LG), np.sin(theta_LG)
    R_LG = np.array(((c, -s), (s, c)))

    position_G = np.dot(R_LG,position_L.T)

    return position_G.T

def occupancygrid_to_numpy(msg):
	data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

	return np.ma.array(data, mask=data==-1, fill_value=-1)