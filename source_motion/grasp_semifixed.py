__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
import math
import os
import sys
# Add the parent folder path to the sys.path list to allow Python to search for modules in specific directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt

class adapt_semi:
    
    def __init__(self, v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, free_axis):
        self.v_0 = v_0
        self.w_0 = w_0
        self.m = m
        self.I = I
        self.f_max_vec = f_max_vec
        self.tau_max_vec = tau_max_vec
        self.v_f = v_f
        self.w_f = w_f
        self.N = N
        self.delta = delta
        self.free_axis = free_axis
    
    # to get the max force (scalar)
    def get_f_max(self):
        f_max = math.sqrt(self.f_max_vec[0]**2 + self.f_max_vec[1]**2 + self.f_max_vec[2]**2)
        return f_max
    
    # to get the max torque (scalar)
    def get_tau_max(self):
        tau_max = math.sqrt(self.tau_max_vec[0]**2 + self.tau_max_vec[1]**2 + self.tau_max_vec[2]**2)
        return tau_max
    
    # to get the final time (scalar)
    def get_t_f(self):
        v_diff = abs(self.v_f - self.v_0)
        w_diff = abs(self.w_f - self.w_0)
        
        if 'x' in self.free_axis: v_diff[0] = 0
        if 'y' in self.free_axis: v_diff[1] = 0
        if 'z' in self.free_axis: v_diff[2] = 0
        
        t_f_v = np.amax(np.divide(v_diff, self.f_max_vec/self.m))
        t_f_w = np.amax(np.divide(w_diff, self.tau_max_vec/self.I))
        t_f = np.amax([t_f_v, t_f_w])
        return t_f
    
    def get_vw_t(self, t, delta_t, v_all, w_all, f_star_all, tau_star_all, f_max, tau_max):
        if t == 0:
            return self.v_0, self.w_0
        else:
            v_last = v_all[-1]
            w_last = w_all[-1]
            f_star_last = f_star_all[-1]
            tau_star_last = tau_star_all[-1]
            v_t = v_last + delta_t * f_star_last / self.m
            w_t = w_last + delta_t * tau_star_last / self.I
            return v_t, w_t
            
    def get_f_star(self, v_t, f_max):
        f_star = - f_max * np.divide((self.m*(v_t - self.v_f)), np.linalg.norm(self.m*(v_t - self.v_f)))
        for i in range(3):
            if abs(v_t[i] - self.v_f[i]) < self.delta: f_star[i] = 0
            if 'x' in self.free_axis: f_star[0] = 0
            if 'y' in self.free_axis: f_star[1] = 0
            if 'z' in self.free_axis: f_star[2] = 0
        return f_star
    
    def get_tau_star(self, w_t, tau_max):
        tau_star = - tau_max * np.divide(self.I*(w_t - self.w_f), np.linalg.norm(self.I*(w_t - self.w_f)))
        for i in range(3):
            if abs(w_t[i] - self.w_f[i]) < self.delta: tau_star[i] = 0
        return tau_star
    
    def pline(self):
        
        t_f = self.get_t_f()
        
        v_all = []
        w_all = []
        f_star_all = []
        tau_star_all = []
        f_max = self.get_f_max()
        tau_max = self.get_tau_max()
        
        for k in range(self.N):
            
            t = k * t_f / self.N
            delta_t = t_f / self.N
            
            # to get the current velocity and angular velocity from last step
            v_t, w_t = self.get_vw_t(t, delta_t, v_all, w_all, f_star_all, tau_star_all, f_max, tau_max)
            v_all.append(v_t)
            w_all.append(w_t)
            
            # to get the current force and torque from current velocity and angular velocity
            # but current force and torque are implemented in the next step
            f_star = self.get_f_star(v_t, f_max)
            tau_star = self.get_tau_star(w_t, tau_max)
            f_star_all.append(f_star)
            tau_star_all.append(tau_star)
            
            # if f_star and tau_star are zero, then the object stops
            if np.linalg.norm(f_star) == 0 and np.linalg.norm(tau_star) == 0:
                t_f = t
                break
        
        f_star_all = np.array(f_star_all)
        tau_star_all = np.array(tau_star_all)
        v_all = np.array(v_all)
        w_all = np.array(w_all)
        
        return t_f, k, f_star_all, tau_star_all, v_all, w_all
    
# ------------------------------------------------------------------------------------
    
class grasp_semifixed(adapt_semi):
    def __init__(self, v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g, free_axis):
        super().__init__(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, free_axis)
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
        w_dot = (tau_star - np.cross(w, self.I*w)) / self.I
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
    

if __name__ == '__main__':
    
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    # When grasping, regard robot and object as the whole
    # Object parameters
    f_max_vec = np.array([10, 10, 10])
    tau_max_vec = np.array([10, 10, 10])
    m = 1.1
    I = 1.5
    N = 2000         # number of steps to iterate through t_f
    delta = 0.01    # to stop when the velocity is close enough to the final velocity
    g = 9.81
    
    # Object initial and final conditions
    v_0 = np.array([3, 5, -2])
    v_f = np.array([0, 0, 0])
    w_0 = np.array([2, -1, 1])
    w_f = np.array([0, 0, 0])
    p_0 = np.array([0, 0, 0])
    q_0 = np.array([1, 0, 0, 0])    # [w, x, y, z]
    free_axis = 'zx'                 # 'x', 'y', 'z', must be lower case
    
    # RK-4 to get the object trajectory
    obj = grasp_semifixed(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g, free_axis)
    t_f, y, p_rk_all, q_rk_all, v_rk_all, w_rk_all, euler_rk_all, v_dot_rk_all, w_dot_rk_all = obj.runge_kutta_method()
    
    print(GREEN + "Final position:          " + RESET, CYAN + str(y[0]) + RESET)
    print(GREEN + "Final quaternion:        " + RESET, CYAN + str(y[1]) + RESET)
    print(GREEN + "Final velocity:          " + RESET, CYAN + str(y[2]) + RESET)
    print(GREEN + "Final angular velocity:  " + RESET, CYAN + str(y[3]) + RESET)
    print(GREEN + "Final time:              " + RESET, CYAN + str(t_f) + RESET)
    
    fig = plt.figure(figsize=(15, 15))  
    ax_1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax_1.plot(p_rk_all[:, 0], p_rk_all[:, 1], p_rk_all[:, 2], color='blue', label='Position trajectory')
    ax_1.set_xlabel('x')
    ax_1.set_ylabel('y')
    ax_1.set_zlabel('z')
    ax_1.scatter(p_rk_all[0, 0], p_rk_all[0, 1], p_rk_all[0, 2], color='red', label='Initial position')
    ax_1.scatter(p_rk_all[-1, 0], p_rk_all[-1, 1], p_rk_all[-1, 2], color='green', label='Final position')
    ax_1.legend()
    
    ax_2 = fig.add_subplot(3, 2, 2, projection='3d')
    ax_2.plot(euler_rk_all[:, 0], euler_rk_all[:, 1], euler_rk_all[:, 2], color='blue', label='Euler angles trajectory')
    ax_2.set_xlabel('roll')
    ax_2.set_ylabel('pitch')
    ax_2.set_zlabel('yaw')
    ax_2.scatter(euler_rk_all[0, 0], euler_rk_all[0, 1], euler_rk_all[0, 2], color='red', label='Initial Euler angles')
    ax_2.scatter(euler_rk_all[-1, 0], euler_rk_all[-1, 1], euler_rk_all[-1, 2], color='green', label='Final Euler angles')
    ax_2.legend()
    
    ax_3 = fig.add_subplot(3, 2, 3, projection='3d')
    ax_3.plot(v_rk_all[:, 0], v_rk_all[:, 1], v_rk_all[:, 2], color='blue', label='Linear velocity trajectory')
    ax_3.set_xlabel('x')
    ax_3.set_ylabel('y')
    ax_3.set_zlabel('z')
    ax_3.scatter(v_rk_all[0, 0], v_rk_all[0, 1], v_rk_all[0, 2], color='red', label='Initial velocity')
    ax_3.scatter(v_rk_all[-1, 0], v_rk_all[-1, 1], v_rk_all[-1, 2], color='green', label='Final velocity')
    ax_3.legend()
    
    ax_4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax_4.plot(w_rk_all[:, 0], w_rk_all[:, 1], w_rk_all[:, 2], color='blue', label='Angular velocity trajectory')
    ax_4.set_xlabel('x')
    ax_4.set_ylabel('y')
    ax_4.set_zlabel('z')
    ax_4.scatter(w_rk_all[0, 0], w_rk_all[0, 1], w_rk_all[0, 2], color='red', label='Initial angular velocity')
    ax_4.scatter(w_rk_all[-1, 0], w_rk_all[-1, 1], w_rk_all[-1, 2], color='green', label='Final angular velocity')
    ax_4.legend()
    
    ax_5 = fig.add_subplot(3, 2, 5, projection='3d')
    ax_5.plot(v_dot_rk_all[:, 0], v_dot_rk_all[:, 1], v_dot_rk_all[:, 2], color='blue', label='Linear acceleration trajectory')
    ax_5.set_xlabel('x')
    ax_5.set_ylabel('y')
    ax_5.set_zlabel('z')
    ax_5.scatter(v_dot_rk_all[0, 0], v_dot_rk_all[0, 1], v_dot_rk_all[0, 2], color='red', label='Initial acceleration')
    ax_5.scatter(v_dot_rk_all[-1, 0], v_dot_rk_all[-1, 1], v_dot_rk_all[-1, 2], color='green', label='Final acceleration')
    ax_5.legend()
    
    ax_6 = fig.add_subplot(3, 2, 6, projection='3d')
    ax_6.plot(w_dot_rk_all[:, 0], w_dot_rk_all[:, 1], w_dot_rk_all[:, 2], color='blue', label='Angular acceleration trajectory')
    ax_6.set_xlabel('x')
    ax_6.set_ylabel('y')
    ax_6.set_zlabel('z')
    ax_6.scatter(w_dot_rk_all[0, 0], w_dot_rk_all[0, 1], w_dot_rk_all[0, 2], color='red', label='Initial angular acceleration')
    ax_6.scatter(w_dot_rk_all[-1, 0], w_dot_rk_all[-1, 1], w_dot_rk_all[-1, 2], color='green', label='Final angular acceleration')
    ax_6.legend()
    
    plt.show() 
