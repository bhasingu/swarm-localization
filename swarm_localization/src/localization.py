#!/usr/bin/env python

'''
    This module has code for the state estimator algorithms
'''

import numpy as np

from helper_functions import *
from swarm import Swarm
from sensor_data import Sensor_Data
import math

class EKF:
    '''
    Class setup for extended kalman filter
    '''
    def __init__(self, T, v_var, w_var, r_var, b_var):

        self.T = T # Time-step, delta(t)
        self.v_var = v_var # Variance of noise added to linear velocity measurements
        self.w_var = w_var # Variance of noise added to angular velocity measurements
        self.r_var = r_var
        self.b_var = b_var
    
    def wraptopi(self, theta):

        return (theta + np.pi) % (2*np.pi) - np.pi

    def motion_model(self, x_k, y_k, theta_k, W_k_v, W_k_w, v_k, w_k):

        X_k_1_hat = np.zeros((3,1))
        X_k_1_hat[0,0] = x_k
        X_k_1_hat[1,0] = y_k
        X_k_1_hat[2,0] = self.wraptopi(theta_k)
        X_k_check = X_k_1_hat + np.matmul(self.T*np.array([[math.cos(theta_k), 0], [math.sin(theta_k), 0], [0, 1]]), np.array([[v_k + W_k_v], [w_k + W_k_w]]))
        X_k_check[2] = self.wraptopi(X_k_check[2])
        
        return X_k_check
    
    def observation_model(self, x_k, y_k, theta_k, N_k_r, N_k_b, l, valid_index):

        num_l = len(valid_index)
        Y_k_check = np.zeros((2*num_l,1))
        for i in range(num_l):
            a_l = l[valid_index[i],0] - x_k
            b_l = l[valid_index[i],1] - y_k
            r_k_l = math.sqrt(pow(a_l,2) + pow(b_l,2)) + N_k_r
            b_k_l = math.atan2(b_l,a_l) - theta_k + N_k_b
            Y_k_check[2*i,0] = r_k_l
            Y_k_check[2*i+1,0] = self.wraptopi(b_k_l)
        
        return Y_k_check
    
    def calc_Q_k(self): # Motion covariance matrix

        Q_k = np.zeros((2,2))
        Q_k[0,0] = self.v_var
        Q_k[1,1] = self.w_var
        
        return Q_k
    
    def calc_F_k_1(self, X_k_1, v_k, w_k): # Motion Jacobian

        F_k_1 = np.zeros((3,3))
        F_k_1[0,0] = 1
        F_k_1[1,1] = 1
        F_k_1[2,2] = 1
        F_k_1[0,2] = (-self.T)*math.sin(X_k_1[2])*v_k
        F_k_1[1,2] = self.T*math.cos(X_k_1[2])*v_k
        
        return F_k_1

    def calc_df_dW(self, X_k_1):

        df_dW = np.zeros((3,2))
        df_dW[0,0] = self.T*math.cos(X_k_1[2])
        df_dW[1,0] = self.T*math.sin(X_k_1[2])
        df_dW[2,1] = self.T
        
        return df_dW
    
    def calc_Q_k_prime(self, df_dW, Q_k):

        Q_k_prime = np.matmul(np.matmul(df_dW, Q_k), np.transpose(df_dW))
        
        return Q_k_prime

    def calc_R_k_l(self, num_l):

        R_k_l = np.zeros((2*num_l,2*num_l))
        for i in range(0,num_l,2):
            R_k_l[i,i] = self.r_var
        for i in range(1,num_l,2):
            R_k_l[i,i] = self.b_var
        
        return R_k_l
    
    def calc_G_k_l(self, x_k, y_k, theta_k, l, valid_index):

        num_l = len(valid_index)
        G_k_l = np.zeros((2*num_l,3))
        for i in range(num_l):
            temp = np.zeros((2,3))
            a_l = l[valid_index[i],0] - x_k
            b_l = l[valid_index[i],1] - y_k
            temp[0,0] = (-a_l)/(math.sqrt(pow(a_l,2) + pow(b_l,2)))
            temp[1,0] = b_l/(pow(a_l,2) + pow(b_l,2))
            temp[0,1] = (-b_l)/(math.sqrt(pow(a_l,2) + pow(b_l,2)))
            temp[1,1] = (-a_l)/(pow(a_l,2) + pow(b_l,2))
            temp[1,2] = -1
            G_k_l[2*i:2*i+2,:] = temp
        
        return G_k_l
    
    def calc_dg_dN(self, num_l):

        return np.identity(2*num_l)

    def calc_R_k_l_prime(self, dg_dN, R_k_l):

        return np.matmul(np.matmul(dg_dN, R_k_l), np.transpose(dg_dN))

    def calc_Y_k(self, valid_index, r_k, b_k):

        num_l = len(valid_index)
        Y_k = np.zeros((2*num_l, 1))
        for i in range(num_l):
            Y_k[2*i,0] = r_k[valid_index[i]]
            Y_k[2*i+1,0] = self.wraptopi(b_k[valid_index[i]])
        
        return Y_k

    def predict(self, X_k_1, P_k_1, v_k, w_k):

        Q_k = self.calc_Q_k()
        df_dW = self.calc_df_dW(X_k_1)
        Q_k_prime = self.calc_Q_k_prime(df_dW, Q_k)
        F_k_1 = self.calc_F_k_1(X_k_1, v_k, w_k)
        P_k_check = np.matmul(np.matmul(F_k_1, P_k_1), np.transpose(F_k_1)) + Q_k_prime
        X_k_check = self.motion_model(X_k_1[0], X_k_1[1], X_k_1[2], 0, 0, v_k, w_k)
        X_k_check[2] = self.wraptopi(X_k_check[2])
        
        return X_k_check.reshape((3)), P_k_check
    
    def update(self, X_k_check, P_k_check, l, valid_index, r_k, b_k):

        num_l = len(valid_index)
        R_k_l = self.calc_R_k_l(num_l)
        dg_dN = self.calc_dg_dN(num_l)
        R_k_l_prime = self.calc_R_k_l_prime(dg_dN, R_k_l)
        G_k_l = self.calc_G_k_l(X_k_check[0], X_k_check[1], X_k_check[2], l, valid_index)
        Y_k = self.calc_Y_k(valid_index, r_k, b_k)
        Y_k_check = self.observation_model(X_k_check[0], X_k_check[1], X_k_check[2], 0, 0, l, valid_index)
        K_k = np.matmul(np.matmul(P_k_check, np.transpose(G_k_l)), np.linalg.pinv(np.matmul(np.matmul(G_k_l, P_k_check), np.transpose(G_k_l)) + R_k_l_prime))
        dim_identity_matrix = len(np.matmul(K_k, G_k_l))
        P_k_hat = np.matmul(np.identity(dim_identity_matrix) - np.matmul(K_k, G_k_l), P_k_check)
        X_k_hat = X_k_check + np.matmul(K_k, Y_k - Y_k_check)
        X_k_hat[2] = self.wraptopi(X_k_hat[2])
        
        return X_k_hat.reshape((3)), P_k_hat