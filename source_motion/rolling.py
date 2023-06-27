__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
import math
from matplotlib import pyplot as plt
import json
import os

class rolling():
    def __init__(self, m, I, tau_max, v_0, w_0, w_max, omega_0, omega_f, p_0, l_c):
        self.m = m
        self.I = I
        self.tau_max = tau_max
        self.v_0 = v_0
        self.w_0 = w_0
        self.w_max = w_max
        self.omega_0 = omega_0
        self.omega_f = omega_f
        self.p_0 = p_0      # the initial position of center of mass
        self.l_c = l_c      # the vertical distance (arm) between center of mass and contact force 
    
    def get_fc(self, omega):
        fc_norm = self.tau_max[0]/self.l_c
        fc_y = fc_norm*math.sin(omega[0])
        fc_z = fc_norm*math.cos(omega[0])
        fc = np.array([0, fc_y, fc_z])
        return fc
    
    def get_linear_acc(self, fc):
        v_acc = fc/self.m + np.array([0, 0, -9.8])  # add GRAVITY!
        return v_acc
    
    def get_angular_acc(self, w):
        w_acc = np.matmul(np.linalg.inv(self.I), self.tau_max) - np.matmul(np.linalg.inv(self.I), np.cross(w, np.matmul(self.I, w)))        # when upgrade to 3-dimension, I is a 3*3 matrix, then * should be replaced by np.matmul
        return w_acc
    
    def pline(self):
        
        w = self.w_0
        omega = self.omega_0
        v = self.v_0
        p = self.p_0
        delta_d = 1e-3
        
        p_all = np.array([p])
        v_all = np.array([v])
        w_all = np.array([w])
        omega_all = np.array([omega])
        
        count = 0
        
        while np.linalg.norm(omega[0] - self.omega_f[0]) > 0.025:
            if np.linalg.norm(w[0] - self.w_max[0]) > 1e-2:
                w_acc = self.get_angular_acc(w)         # angular acceleration at current time
                fc = self.get_fc(omega)                 # contact force at current time
            else:
                w_acc = 0
                fc = np.array([0, 0, 0])
            v_acc = self.get_linear_acc(fc)             # linear acceleration at current time
            
            omega += w*delta_d + 0.5*w_acc*delta_d**2   # omega at next time
            w += w_acc*delta_d                          # angular velocity at next time
            
            p += v*delta_d + 0.5*v_acc*delta_d**2       # position at next time
            v += v_acc*delta_d                          # linear velocity at next time
            
            p_all = np.vstack((p_all, p))
            v_all = np.vstack((v_all, v))
            w_all = np.vstack((w_all, w))
            omega_all = np.vstack((omega_all, omega))
            
        return p_all, v_all, w_all, omega_all
            
    
    
if __name__ == '__main__':
    
    # Only consider the one axis rotation
    m = 1
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])   
    tau_max = np.array([7, 0, 0])
    
    v_0 = np.array([0.0, 2.0, 0.0])
    w_0 = np.array([0.3, 0, 0])
    w_max = np.array([math.pi/1.9, 0, 0])       # constrained by max angular velocity
    omega_0 = np.array([0.0, 0.0, 0.0])
    omega_f = np.array([math.pi, 0, 0])       # rotate around x axis  
    p_0 = np.array([0.0, 0.0, 4.0])
    l_c = 0.15
    
    # contact point parameters (initial status is parallel to the ground)
    # Clockwise is positive direction
    # -----------------------
    # |         o           |
    # -----c-----------------
    #      
    # ^:z ->:y /:x
    # c: contact point  o: center of mass
    # length: 0.4m      width: 0.1m
    # c = o + [0, 0.1, 0.05] initial trans 
    # ASSUME the contact force is always vertical to edge of contact point
    # Set object rotate around center of mass instead of contact point

    roll = rolling(m, I, tau_max, v_0, w_0, w_max, omega_0, omega_f, p_0, l_c)
    p_all, v_all, w_all, omega_all = roll.pline()
    
    roll_file_path = os.path.join(os.getcwd(), 'data/roll_data')
    data_dict = {'Position': np.matrix(p_all).tolist(), 'Orientation': np.matrix(omega_all).tolist()}
    json_object = json.dumps(data_dict, indent = 4)
    with open(roll_file_path, 'w') as outfile: outfile.write(json_object)
    
    
    # plot the trajectory
    fig = plt.figure(figsize=(15, 15))
    
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(p_all[:, 0], p_all[:, 1], p_all[:, 2], label='Position')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.legend()
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(omega_all[:, 0], omega_all[:, 1], omega_all[:, 2], label='Orientation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.legend()
    
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(v_all[:, 0], v_all[:, 1], v_all[:, 2], label='Linear Velocity')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.legend()
    
    ax4 = fig.add_subplot(224)
    ax4.plot(w_all[:, 0], label='Angular Velocity')
    ax4.set_xlabel('time')
    ax4.set_ylabel('Angular Velocity')
    ax4.legend()
    
    plt.show()