__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
np.seterr(invalid='ignore')
import math
from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.animation as animation
import json
import os 

class sticking():
    def __init__(self, m, I, tau_max, F_max, v_0, w_0, w_max, omega_0, omega_f, p_0, l):
        self.m = m
        self.I = I
        self.tau_max = tau_max
        self.F_max = F_max
        self.v_0 = v_0
        self.w_0 = w_0
        self.w_max = w_max
        self.omega_0 = omega_0
        self.omega_f = omega_f
        self.p_0 = p_0      # the initial position of center of mass
        self.l = l      # the vertical distance (arm) between center of mass and contact force 
    
    def get_tau(self, omega):
        numerator = np.matmul(self.I, omega - self.omega_f)
        denominator = np.abs(numerator)
        sign = np.divide(numerator, denominator)
        tau = -sign[0]*self.tau_max     # only consider the first dimension, since it's 2d rotation
        return tau
     
    def get_angular_acc(self, w, tau):
        w_acc = np.matmul(np.linalg.inv(self.I), tau - np.cross(w, np.matmul(self.I, w)))        # when upgrade to 3-dimension, I is a 3*3 matrix, then * should be replaced by np.matmul
        return w_acc
    
    def get_f_diff(self, tau):
        f_diff = 2*tau/self.l
        return f_diff[0]
    
    def func(self, F_1, f_diff, omega):
        # given F_1 - F_2, omega, to make below function maximum
        # cos is to make z direction fastest, sin is to make y direction fastest
        return (2*F_1 - f_diff) * math.sin(omega[0])
    
    def get_F(self, f_diff, omega):
        objective_fast = lambda F_1: -self.func(F_1, f_diff, omega)
        objective_slow = lambda F_1: self.func(F_1, f_diff, omega)
        # Bound is F_1, F_2 in range [0, F_max], also F_z = (F_1 + F_2)*cos(omega)>mg
        bound_min = max(0, f_diff, 0.5*(f_diff + self.m*9.72/math.cos(omega[0])))
        bound_max = min(self.F_max, f_diff + self.F_max, 0.5*(f_diff + self.m*20/math.cos(omega[0])))
        result_fast = minimize_scalar(objective_fast, bounds=(bound_min, bound_max), method='bounded')
        result_slow = minimize_scalar(objective_slow, bounds=(bound_min, bound_max), method='bounded')
        fast_weight = 0.075
        optimal_F_1 = result_fast.x * fast_weight + result_slow.x * (1 - fast_weight)
        optimal_F_2 = optimal_F_1 - f_diff
        return optimal_F_1, optimal_F_2
        
    def get_linear_acc(self, fc):
        v_acc = fc/self.m + np.array([0, 0, -9.8])  # add GRAVITY!
        return v_acc
    
    def pline(self):
        
        w = self.w_0
        omega = self.omega_0
        v = self.v_0
        p = self.p_0
        F_1 = 0
        F_2 = 0
        F = np.array([F_1, F_2]) # F = (F_1, F_2)
        delta_d = 1e-3
        
        p_all = np.array([p])
        v_all = np.array([v])
        w_all = np.array([w])
        F_all = np.array([F])
        omega_all = np.array([omega])
        
        count = 0
        
        while np.linalg.norm(omega[0] - self.omega_f[0]) > 0.025:
                    
            if np.linalg.norm(np.abs(w[0]) - self.w_max[0]) > 1e-2:
                tau = self.get_tau(omega)                       # get the tau at current time
                w_acc = self.get_angular_acc(w, tau)            # angular acceleration at current time
                f_diff = self.get_f_diff(tau)                   # get the F_1 - F_2 
            else:
                w_acc = 0
                f_diff = 0
            F_1, F_2 = self.get_F(f_diff, omega)               # get the F_1 and F_2
            F_y = (F_1 + F_2)*math.sin(omega[0])               # get the F_y
            F_z = (F_1 + F_2)*math.cos(omega[0])               # get the F_z
            
            v_acc = self.get_linear_acc(np.array([0, F_y, F_z]))   # linear acceleration at current time
            
            omega += w*delta_d + 0.5*w_acc*delta_d**2   # omega at next time
            w += w_acc*delta_d                          # angular velocity at next time
            
            p += v*delta_d + 0.5*v_acc*delta_d**2       # position at next time
            v += v_acc*delta_d                          # linear velocity at next time
            
            p_all = np.vstack((p_all, p))
            v_all = np.vstack((v_all, v))
            w_all = np.vstack((w_all, w))
            omega_all = np.vstack((omega_all, omega))
            F_all = np.vstack((F_all, np.array([F_1, F_2])))
                
            
        return p_all, v_all, w_all, omega_all, F_all
            
    
    
if __name__ == '__main__':
    
    # Only consider the one axis rotation
    m = 1
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    F_max = 10.15    # the maximum of F_1 and F_2
    l = 0.3  
    tau_max = np.array([F_max*l/2, 0, 0])
    w_max = np.array([math.pi/3, 0, 0])       # constrained by max angular velocity
    
    v_0 = np.array([0, 0.2, -1.0])
    w_0 = np.array([0.15, 0, 0])
    omega_0 = np.array([0.0, 0.0, 0.0])
    omega_f = np.array([math.pi/10, 0, 0])     # rotate around x axis  
    p_0 = np.array([0.2, 0.3, 3])
    
    # contact point parameters (initial status is parallel to the ground)
    # Clockwise is positive direction
    # -----------------------
    # |         o           |
    # -----F1---------F2-----
    #      
    # ^:z ->:y /:x
    # Sticking in y-z plane

    stick1 = sticking(m, I, tau_max, F_max, v_0, w_0, w_max, omega_0, omega_f, p_0, l)
    p_all_1, v_all_1, w_all_1, omega_all_1, F_all_1 = stick1.pline()
    
    v_0_2 = v_all_1[-1]
    w_0_2 = np.array([0.0, 0.0, 0.0])
    omega_0_2 = omega_all_1[-1]
    omega_f_2 = np.array([-math.pi/10, 0, 0])
    p_0_2 = p_all_1[-1]
    
    stick2 = sticking(m, I, tau_max, F_max, v_0_2, w_0_2, w_max, omega_0_2, omega_f_2, p_0_2, l)
    p_all_2, v_all_2, w_all_2, omega_all_2, F_all_2 = stick2.pline()
    
    v_0_3 = v_all_2[-1]
    w_0_3 = np.array([0.0, 0.0, 0.0])
    omega_0_3 = omega_all_2[-1]
    omega_f_3 = np.array([0, 0, 0])
    p_0_3 = p_all_2[-1]
    
    stick3 = sticking(m, I, tau_max, F_max, v_0_3, w_0_3, w_max, omega_0_3, omega_f_3, p_0_3, l)
    p_all_3, v_all_3, w_all_3, omega_all_3, F_all_3 = stick3.pline()
    
    p_all = np.vstack((p_all_1[:-1], p_all_2[:-1], p_all_3))
    omega_all = np.vstack((omega_all_1[:-1], omega_all_2[:-1], omega_all_3))
    v_all = np.vstack((v_all_1[:-1], v_all_2[:-1], v_all_3))
    w_all = np.vstack((w_all_1, w_all_2, w_all_3))
    F_all = np.vstack((F_all_1, F_all_2, F_all_3))
    
    stick_file_apth = os.path.join(os.getcwd(), 'data/sticking_data')
    data_dict = {'Position':np.matrix(p_all).tolist(), 'Orientation':np.matrix(omega_all).tolist()}
    json_object = json.dumps(data_dict, indent = 4)
    with open(stick_file_apth, 'w') as outfile: outfile.write(json_object)
    
    # plot the trajectory
    fig = plt.figure(figsize=(15, 15))
    
    ax1 = fig.add_subplot(321, projection='3d')
    ax1.plot(p_all[:, 0], p_all[:, 1], p_all[:, 2], label='Position')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.legend()
    
    ax2 = fig.add_subplot(322, projection='3d')
    ax2.plot(omega_all[:, 0], omega_all[:, 1], omega_all[:, 2], label='Orientation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.legend()
    
    ax3 = fig.add_subplot(323, projection='3d')
    ax3.plot(v_all[:, 0], v_all[:, 1], v_all[:, 2], label='Linear Velocity')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.legend()
    
    ax4 = fig.add_subplot(324)
    ax4.plot(w_all[:, 0], label='Angular Velocity')
    ax4.set_xlabel('time')
    ax4.set_ylabel('Angular Velocity')
    ax4.legend()
    
    ax5 = fig.add_subplot(325)
    ax5.plot(F_all[:, 0], color='r', label='F_1')
    ax5.plot(F_all[:, 1], color='g', label='F_2')
    ax5.set_xlabel('time')
    ax5.set_ylabel('Force')
    ax5.legend()
    
    plt.show()
    
    
    # Dynamic plot
    # def update(frame):
    #     plot1.set_data(p_all[:frame,0], p_all[:frame,1])
    #     plot1.set_3d_properties(p_all[:frame,2])
        
    #     plot2.set_data(omega_all[:frame,0], omega_all[:frame,1])
    #     plot2.set_3d_properties(omega_all[:frame,2])
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    
    # plot1 = ax1.plot([], [], [], 'r', label='Position')[0]
    # plot2 = ax2.plot([], [], [], 'b', label='Orientation')[0]
    
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    # ax1.set_title('Sticking Trajectory')
    
    # ax2.set_xlabel('roll')
    # ax2.set_ylabel('pitch')
    # ax2.set_zlabel('yaw')
    # ax2.set_title('Sticking Altitude')
    # ax2.legend()
    
    # ani = animation.FuncAnimation(fig, update, frames=len(p_all), interval=10, repeat=False, blit=False)
    # ani.save('sticking', writer='pillow')