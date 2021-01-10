#!/usr/bin/env python

'''
    This module handles landmark detection for the localization of the swarm
'''

import numpy as np
import itertools
import math

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose2D, Pose, Quaternion
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sklearn import mixture
from sklearn.linear_model import LinearRegression
import helper_functions as helper

def groupCluster(pcd):
    '''
    This function clusters point-cloud data into 2 groups (each group representing an edge in the environment)
    Input:
        pcd - Point-cloud; datatype: Open3D pointcloud
    Output:
        ptc - Pointcloud cluster; datatype: np.array [num_cluster, num_point, 2]
    '''
    # Label for each point in pointcloud
    # eps is the min distance for point in same cluster
    labels = pcd.cluster_dbscan(eps = 0.3, min_points = 3, print_progress=False)
    labels = np.asarray(labels)
    num = 0
    ptc = []
    cluster = []
    pcd = np.asarray(pcd.points)
    for i, p in enumerate(pcd):
        if labels[i] == -1:
            # A group label for a pointcloud of -1 indicates that it's noise and couldn't be assigned to a group
            continue
        if labels[i] > num:
            ptc.append(np.array(cluster))
            cluster = []
            num += 1
        cluster.append(p)
    ptc.append(np.array(cluster))
    return np.array(ptc)

def my_round(x, prec=2, base=0.1):
    return (base * (np.array(x) / base).round()).round(prec)

def HoughLines(pcd):
    '''
    Extract lines from clustered point-cloud using Hougg Line Transform.
    Need to set threshold for how many vote for a line extracted to be considered as a candidate.
    Input:
        Numpy num_point x 2 : pointcloud cluster
    Output: 
        {(rho, theta) : num_vote} dictionary : Lines
    '''
    lines = {}
    threshold = 5
    point_set = set()
    # loop through all lines and update vote
    for i, p1 in enumerate(pcd):
        for j, p2 in enumerate(pcd):
            if j <= i:
                continue
            # theta = np.around(np.arctan2((f[0]-s[0]),(s[1]-f[1])),1)
            theta = np.arctan((p1[0]-p2[0])/(p2[1]-p1[1]))
            # if theta <= 0:
            #     theta = theta + np.pi
            theta = my_round(theta)
            rho = my_round(p1[0] * np.cos(theta) + p1[1] * np.sin(theta))
            # if rho < 0:
            #     continue
            if (rho, theta) in lines.keys():
                lines[( rho, theta)][0] += 1
                lines[( rho, theta)][1].add(tuple(p1))
                lines[( rho, theta)][1].add(tuple(p2))
            else:
                lines[( rho, theta)] = [1, {tuple(p1), tuple(p2)}]

    # remove vote below threshold
    for pair in lines.items():
        if pair[1][0] < threshold:
            lines.pop(pair[0])
        else:
            for pt in pair[1][1]:
                point_set.add(tuple(pt))
    if len(lines) >= 2:
        line1 = []
        line2 = []
        X = np.array(list(point_set))
        gmm = mixture.GaussianMixture(n_components=2,
                                      covariance_type='diag')
        gmm.fit(X)
        y = gmm.predict(X)
        for i in range(len(y)):
            if y[i] == 0:
                line1.append(X[i])
            else:
                line2.append(X[i])
        line1 = np.array(line1)
        line2 = np.array(line2)
        print("here")
        print(line1)
        l1 = regression(line1)
        print(line2)
        l2 = regression(line2)
        if (line1 is None) or (line2 is None) or (l1 is None) or (l2 is None):
            return lines
        l1.update(l2)
        return l1
    
    return lines

def regression(line):
    # input: points of line
    # output: line equation (rho, theta)
    if len(line) < 2:
        return None
    X = line[:,0]
    y = line[:,1]
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)

    reg = LinearRegression().fit(X,y)
    print("slope: ", reg.coef_)
    print("b: ", reg.intercept_)
    print("score", reg.score(X,y))
    
    theta = np.arctan(-1/reg.coef_)
    rho = reg.intercept_ * np.sin(theta)
    MRE = meanrhoerror(X,y,rho, theta)
    print("rho: ", rho)
    print("theta: ", theta)
    print("MRE: " , MRE)

    flipreg = LinearRegression().fit(y,X)
    print("fslope: ", flipreg.coef_)
    print("fb: ", flipreg.intercept_)
    print("fscore", flipreg.score(y,X))
    
    ftheta = np.arctan(-1/flipreg.coef_)
    frho = flipreg.intercept_ * np.sin(ftheta)
    ftheta = helper.wraptopi(np.pi - (ftheta + np.pi/2))
    flipMRE = meanrhoerror(X,y,frho,ftheta)
    print("frho: ", frho)
    print("ftheta: ", ftheta)
    print("flipMRE: ", flipMRE)

    if MRE < flipMRE:
        return {(float(rho), float(theta)) : 1}
    else:
        return {(float(frho), float(ftheta)) : 1}

def meanrhoerror(x,y,rho,theta):
    total = 0
    for i in range(len(x)):
        pt_rho = x[i] * np.cos(theta) + y[i] * np.sin(theta)
        error = abs(rho - pt_rho)
        total += error
    meanerror = total / len(x)
    return meanerror

def findIntersect(lines):
    '''
    Find the intersection of the lines which could be where landmark is
    Input: {(rho, theta) : num_vote} dictionary : Lines
    Output: Numpy num_intersect x 2 : line intersection
    '''
    intersect = []
    rho_dist_threshold = 0.1
    angle_threshold = 0.1
    for i, l1 in enumerate(lines.keys()):
        for j, l2 in enumerate(lines.keys()):
            if j <= i:
                continue
            if abs(abs(l1[0]) - abs(l2[0])) < rho_dist_threshold and (abs(abs(l1[1]) - abs(l2[1])) < angle_threshold):
                continue
            rho_vect = np.array([l1[0], l2[0]])
            theta_vect = np.array([[np.cos(l1[1]), np.sin(l1[1])],[np.cos(l2[1]), np.sin(l2[1])]])
            try:
                inv = np.linalg.inv(theta_vect)
                ans = np.matmul(inv,rho_vect.T)
            except:
                continue
            intersect.append(ans)
    return np.array(intersect)

def concat_potential_lm(intersect, potential_lm):
    '''
    Concatenate the intersection and potential_landmark where the cluster change
    Input: Line Intersection
           Potential Landmark
    Output: num_potential landmark x 2
    '''
    if intersect.size != 0 and potential_lm.size != 0:
        return np.concatenate((intersect, potential_lm))
    if potential_lm.size == 0:
        return intersect
    if intersect.size == 0:
        return potential_lm

def checkNearestLandmark(intersect, landmark_pos):

    # dist_threshold = 0.05
    dist_threshold = 0.15

    closest_pt2lm = np.zeros(np.array(landmark_pos).shape)
    min_dist = 10000
    min_intercept = 0
    min_landmark_idx = 0
    for i, pt_lm in enumerate(intersect):
        for j, true_lm in enumerate(landmark_pos):
            dist = np.sqrt((pt_lm[0] - true_lm[0]) ** 2 + (pt_lm[1] - true_lm[1])**2)
            if dist < dist_threshold and dist < min_dist:
                min_dist = dist
                min_intercept = pt_lm
                min_landmark_idx = j
    closest_pt2lm[min_landmark_idx] = min_intercept
    return closest_pt2lm

def robot2seen_lm(robot_config, closest_pt2lm):
    r2lm = np.zeros((len(closest_pt2lm),2))
    valid_index = []
    for i, lm in enumerate(closest_pt2lm):
        if np.any(lm):
            dist = np.sqrt((robot_config[0] - lm[0])**2 + (robot_config[1] - lm[1])**2)
            print("dist")
            print(dist)
            lm_angle = np.arctan2(lm[1] - robot_config[1] , lm[0] - robot_config[0]) - wraptopi(robot_config[2])
            # angle = robot_config[2] - lm_angle
            r2lm[i,0] = dist
            r2lm[i,1] = wraptopi(lm_angle)
            valid_index.append(i)
            # r2lm.append([dist, lm_angle])
        else:
            # r2lm.append([0,0])
            r2lm[i,0] = 0.0
            r2lm[i,1] = 0.0
    return r2lm, valid_index

def convert2world(x, y, C_t):
    '''
    Transform the x and y point of all robot frame to world frame
    Inputs:
        x - num_robot x num_lidarscan , x coordinate of sensor data from robot
        y - num_robot x num_lidarscan , y coordinate of sensor data from robot
        C_t - robots' configuration, need this to transform back to world frame
    Outputs:
        all_point - num_robot x num_lidarscan x 2 np.array
    '''
    all_point = []
    for num_rob in range(len(C_t)):
        data_point = []
        for num_data in range(len(x[num_rob])):
            data_point.append([x[num_rob][num_data], y[num_rob][num_data]])
        data_point = np.array(data_point)
        rotate_data = helper.rotate(data_point, C_t[num_rob][2])
        rotate_data = np.add(rotate_data, C_t[num_rob][:2])
        z = np.zeros((len(x[num_rob]),3))
        z[:,:-1] = rotate_data
        all_point.append(z)
    return np.array(all_point)

def motion_model(x_k_1, y_k_1, theta_k_1, v_k, w_k, W_k_v, W_k_w, T):
    '''
    This function uses the motion model to estimate the robot position after motion using odometry information
    '''
    X_k_1_hat = np.zeros((3,1))
    X_k_1_hat[0,0] = x_k_1
    X_k_1_hat[1,0] = y_k_1
    X_k_1_hat[2,0] = helper.wraptopi(theta_k_1)
    X_k_check = X_k_1_hat + np.matmul(T*np.array([[math.cos(theta_k_1), 0], [math.sin(theta_k_1), 0], [0, 1]]), np.array([[v_k + W_k_v], [w_k + W_k_w]]))
    X_k_check[2] = helper.wraptopi(X_k_check[2])
    return X_k_check

def point_based_matching(P, Q):
    '''
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.
    P - Laser Scan data from robot represented in global frame
    Q - Associated data points from known map
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    '''
    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = np.shape(P)[0]
    print("n = " + str(n))

    # Safety check
    if n == 0:
        return None, None, None

    for i in range(n):

        x = P[i, 0]
        y = P[i, 1]
        xp = Q[i, 0]
        yp = Q[i, 1]

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for i in range(n):

        x = P[i, 0]
        y = P[i, 1]
        xp = Q[i, 0]
        yp = Q[i, 1]

        s_x_xp += (x - x_mean)*(xp - xp_mean)
        s_y_yp += (y - y_mean)*(yp - yp_mean)
        s_x_yp += (x - x_mean)*(yp - yp_mean)
        s_y_xp += (y - y_mean)*(xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
    translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

    return rot_angle, translation_x, translation_y