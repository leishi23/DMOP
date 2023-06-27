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
from source_function.forward_dynamics import forward_dynamics
from matplotlib import pyplot as plt
import json


class grasp_fixed(forward_dynamics):
    def __init__(self, v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g):
        super().__init__(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g)
    
    def res(self):
        return self.runge_kutta_method()
    
    
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
    f_max_vec = np.array([3, 3, 3])
    tau_max_vec = np.array([3, 3, 3])
    m = 1.1
    I = np.eye(3)
    N = 200         # number of steps to iterate through t_f
    delta = 0.01    # to stop when the velocity is close enough to the final velocity
    g = 9.81
    
    # Object initial and final conditions
    v_0 = np.array([3, 4, -0.5])
    v_f = np.array([0, 0, 0])
    w_0 = np.array([0.1, 4, 0.1])
    w_f = np.array([0, 0, 0])
    p_0 = np.array([0, 0, 5])
    q_0 = np.array([1, 0, 0, 0])    # [w, x, y, z]
    
    # RK-4 to get the object trajectory
    obj = grasp_fixed(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta, p_0, q_0, g)
    t_f, y, p_rk_all, q_rk_all, v_rk_all, w_rk_all, euler_rk_all, v_dot_rk_all, w_dot_rk_all = obj.res()
    
    stick_file_apth = os.path.join(os.getcwd(), 'data/grasp_fixed_data')
    data_dict = {'Position':np.matrix(p_rk_all).tolist(), 'Orientation':np.matrix(euler_rk_all).tolist()}
    json_object = json.dumps(data_dict, indent = 4)
    with open(stick_file_apth, 'w') as outfile: outfile.write(json_object)
    
    print(GREEN + "Final position:          " + RESET, CYAN + str(y[0]) + RESET)
    print(GREEN + "Final quaternion:        " + RESET, CYAN + str(y[1]) + RESET)
    print(GREEN + "Final velocity:          " + RESET, CYAN + str(y[2]) + RESET)
    print(GREEN + "Final angular velocity:  " + RESET, CYAN + str(y[3]) + RESET)
    print(GREEN + "Final time:              " + RESET, CYAN + str(t_f) + RESET)
    
    fig = plt.figure(figsize=(15, 10))  
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