# %%
__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"

import numpy as np
import math
import os
import sys
from matplotlib import pyplot as plt
from source_function.adaptation import adapt

# Input: initial state, mass, inertia, maximum force, maximum torque, ideal final velocity, ideal final angular velocity, number of steps, delta, gravity
# Output: final time, final state([position/orientation/velocity/angular velocity]), state at each time step

class forward_dynamics(adapt):
    def __init__(self, v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g):
        super().__init__(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta)
        self.t_f = self.pline()[0]
        self.k = self.pline()[1]    # how many time steps really used
        self.f_star_all = self.pline()[2]
        self.tau_star_all = self.pline()[3]
        self.p_0 = p_0
        self.q_0 = q_0
        self.v_0 = v_0
        self.w_0 = w_0
        self.m = m
        self.I = I
        self.g = g
     
    # Input: current force and torque, current angular velocity
    # Output: current linear acceleration and angular acceleration   
    def euler_newton_eq(self, f_star, tau_star, w): # w here is not w_star but from the previous step(Ruge-Kutta)
        # v_dot = (f_star - self.m*self.g) / self.m
        v_dot = f_star / self.m
        # w_dot = (tau_star - np.cross(w, np.matmul(self.I, w))) / self.I
        w_dot = np.matmul(np.linalg.inv(self.I), tau_star - np.matmul(np.cross(w, self.I), w))
        return v_dot, w_dot
    
    def quart_dot_w(self, qk, wk): # use hamiltonian equation to get the derivative of the quaternion
        hamiltonian = np.array([[0, -wk[0], -wk[1], -wk[2]], 
                                [wk[0], 0, wk[2], -wk[1]], 
                                [wk[1], -wk[2], 0, wk[0]], 
                                [wk[2], wk[1], -wk[0], 0]])
        return 0.5 * np.dot(hamiltonian, qk)
    
    # Input: current time, current state
    # Output: the derivative of the state
    def dynamics_eq(self, count, t, y):
        p = y[0]
        q = y[1]    # unit quaternion, represent the orientation of the body frame
        v = y[2]
        w = y[3]
        f_star = self.f_star_all[count]
        tau_star = self.tau_star_all[count]
        
        p_dot = v
        q_dot = self.quart_dot_w(q, w)
        v_dot, w_dot = self.euler_newton_eq(f_star, tau_star, w)
        
        return np.array([p_dot, q_dot, v_dot, w_dot], dtype=object)
    
    def euler_from_quaternion(self, q):
    # assumes q = [q0, q1, q2, q3] (scalar first)
        dcm = self.dcm_from_quaternion(q)

        psi = np.arctan2(dcm[0, 1], dcm[0, 0])  # yaw
        theta = np.arcsin(-dcm[0, 2])  # pitch
        phi = np.arctan2(dcm[1, 2], dcm[2, 2])  # roll

        return np.array([psi, theta, phi])
    
    def dcm_from_quaternion(self, q):
        q0, q1, q2, q3 = q  # [0],q[1],q[2],q[3]

        return np.array([
            [2 * q0 ** 2 - 1 + 2 * q1 ** 2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), 2 * q0 ** 2 - 1 + 2 * q2 ** 2, 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 2 * q0 ** 2 - 1 + 2 * q3 ** 2]
        ])
    
    def runge_kutta_method(self):
        # Runge-Kutta 4th Order Method 
        t = 0
        count = 0
        y = np.array([self.p_0, self.q_0, self.v_0, self.w_0], dtype=object)
        h = (self.t_f - t) / self.k  # step size
        
        p_rk_all = [self.p_0]
        q_rk_all = [self.q_0]
        v_rk_all = [self.v_0]
        w_rk_all = [self.w_0]
        v_dot_rk_all = [self.dynamics_eq(count, t, y)[2]]
        w_dot_rk_all = [self.dynamics_eq(count, t, y)[3]]
        euler_rk_all = [self.euler_from_quaternion(self.q_0)]
        
        while t <= self.t_f and abs(t - self.t_f) > 1e-4:
            count += 1
            k1 = h * self.dynamics_eq(count, t, y)
            k2 = h * self.dynamics_eq(count, t + h/2, y + k1/2)
            k3 = h * self.dynamics_eq(count, t + h/2, y + k2/2)
            k4 = h * self.dynamics_eq(count, t + h, y + k3)
            y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t = t + h
            p_rk_all.append(y[0])
            q_rk_all.append(y[1])
            v_rk_all.append(y[2])
            w_rk_all.append(y[3])
            v_dot_rk_all.append(self.dynamics_eq(count, t, y)[2])
            w_dot_rk_all.append(self.dynamics_eq(count, t, y)[3])
            euler_rk_all.append(self.euler_from_quaternion(y[1]))
        
        return self.t_f, y, np.array(p_rk_all), np.array(q_rk_all), np.array(v_rk_all), np.array(w_rk_all), np.array(euler_rk_all), np.array(v_dot_rk_all), np.array(w_dot_rk_all)
    
    # %% 
if __name__ == '__main__': # when running cell, comment this line

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    # object parameters
    f_max_vec = np.array([1, 2, 3])         # maximum force that each thruster can provide
    tau_max_vec = np.array([2, 1, 3])       # maximum torque that each thruster can provide
    m = 2                                   # mass of the object
    I = 3                                   # inertia of the object
    N = 1000                                 # number of steps
    delta = 0.01                            # to stop when the velocity is close enough to the final velocity
    g = 9.81
    
    # Initial and final conditions
    v_0 = np.array([2, 3, -2])
    v_f = np.array([-0.5, -1, 1.5])
    w_0 = np.array([0, 0, 0])
    w_f = np.array([2, -1, 1])
    p_0 = np.array([0, 0, 0])
    q_0 = np.array([1, 0, 0, 0])    # [w, x, y, z]
    
    # %%
    
    forward_dynamics_obj = forward_dynamics(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g)
    t_f, y, p_rk_all, q_rk_all, v_rk_all, w_rk_all, euler_rk_all, v_dot_rk_all, w_dot_rk_all = forward_dynamics_obj.runge_kutta_method()
    print(GREEN + "Final position:          " + RESET, CYAN + str(y[0]) + RESET)
    print(GREEN + "Final quaternion:        " + RESET, CYAN + str(y[1]) + RESET)
    print(GREEN + "Final velocity:          " + RESET, CYAN + str(y[2]) + RESET)
    print(GREEN + "Final angular velocity:  " + RESET, CYAN + str(y[3]) + RESET)
    print(GREEN + "Final time:              " + RESET, CYAN + str(t_f) + RESET)
    
    # plot the position,  velocity, angular velocity
    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(p_rk_all[:, 0], p_rk_all[:, 1], p_rk_all[:, 2], label='position')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(v_rk_all[:, 0], v_rk_all[:, 1], v_rk_all[:, 2], label='linear velocity')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(w_rk_all[:, 0], w_rk_all[:, 1], w_rk_all[:, 2], label='angular velocity')
    ax3.legend()
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot(euler_rk_all[:, 0], euler_rk_all[:, 1], euler_rk_all[:, 2], label='euler angles')
    ax4.legend()
    ax4.set_xlabel('yaw')
    ax4.set_ylabel('pitch')
    ax4.set_zlabel('roll')
    
    plt.show()
