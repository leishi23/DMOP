__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"

import numpy as np
import math 
from matplotlib import pyplot as plt
 
# Input: initial velocity v_0, initial angular velocity w_0, mass m, inertia I, max force vector f_max_vec, max torque vector tau_max_vec,
#        final velocity v_f, final angular velocity w_f
# Output: final time t_f, list of force f_star, list of torque tau_star

class adapt:
    
    def __init__(self, v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta) -> None:
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
            w_t = w_last + delta_t * np.matmul(np.linalg.inv(self.I), tau_star_last)
            return v_t, w_t
            
    def get_f_star(self, v_t, f_max):
        f_star = - f_max * np.divide((self.m*(v_t - self.v_f)), np.linalg.norm(self.m*(v_t - self.v_f)))
        for i in range(3):
            if abs(v_t[i] - self.v_f[i]) < self.delta: f_star[i] = 0
        return f_star
    
    def get_tau_star(self, w_t, tau_max):
        tau_star = - tau_max * np.divide(np.matmul(self.I, (w_t - self.w_f)), np.linalg.norm(self.I*(w_t - self.w_f)))
        for i in range(3):
            if abs(w_t[i] - self.w_f[i]).all() < self.delta: tau_star[i] = 0
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
        
        f_star_all = np.array(f_star_all, dtype=object)
        tau_star_all = np.array(tau_star_all, dtype=object)
        v_all = np.array(v_all, dtype=object)
        w_all = np.array(w_all, dtype=object)
        
        return t_f, k, f_star_all, tau_star_all, v_all, w_all
            
    
if __name__ == "__main__":
    
    # object parameters
    f_max_vec = np.array([1, 2, 3])
    tau_max_vec = np.array([2, 1, 3])
    m = 2
    I = 3
    N = 200
    delta = 0.01    # to stop when the velocity is close enough to the final velocity
    
    # Initial and final conditions
    v_0 = np.array([1, 2, -1])
    v_f = np.array([-3, -3, 4])
    w_0 = np.array([0, 0, 0])
    w_f = np.array([2, -1, 1])
    
    adapt_obj = adapt(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta)
    t_f, k, f_star_all, tau_star_all, v_all, w_all = adapt_obj.pline()
    
    print('t_f: ', t_f)
    # print('f_star_all: ', f_star_all)
    # print('tau_star_all: ', tau_star_all)
    
    ## plot the force and torque
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(v_all[:, 0], v_all[:, 1], v_all[:, 2], 'red', label='velocity')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.scatter(v_0[0], v_0[1], v_0[2], c='black', marker='o')
    ax1.scatter(v_f[0], v_f[1], v_f[2], c='black', marker='o')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(w_all[:, 0], w_all[:, 1], w_all[:, 2], 'blue', label='angular velocity')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.scatter(w_0[0], w_0[1], w_0[2], c='black', marker='o')
    ax2.scatter(w_f[0], w_f[1], w_f[2], c='black', marker='o')
    
    plt.show()