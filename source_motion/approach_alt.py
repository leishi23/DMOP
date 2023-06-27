__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
import math 
from matplotlib import pyplot as plt
import os
import sys
# Add the parent folder path to the sys.path list to allow Python to search for modules in specific directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source_function.parabola import parabola
import matplotlib.animation as animation

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Input: object's initial position/orientation/angular velocity, any given time tf_K, mass, inertia, torque_max, force_max, iterate time
# Output: object and robot orientation list of each step

class robot_attitude(parabola):
    def __init__(self, w0, I, tau_max, omega0, d_t, x0, y0, z0, v0, a, d_time, w1, w2, t0, robo_omega0):
        super().__init__(w0, I, tau_max, omega0, d_t, x0, y0, z0, v0, a, d_time)
        self.w1 = w1
        self.w2 = w2
        self.t0 = t0
        self.tf = d_time    # final time
        self.obj_omegaf = self.predict()[1]
        self.obj_wf = self.predict()[3]
        self.robo_omega0 = robo_omega0
        self.robo_omegaf = self.predict()[1]
        self.robo_wf = 0    # set 0, it's OK 
    
    # (roll, pitch ,yaw）转换为3*3旋转矩阵
    def euler2rotm(self, euler):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(euler[0]), -np.sin(euler[0])],
                        [0, np.sin(euler[0]), np.cos(euler[0])]
                        ])
        R_y = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])],
                        [0, 1, 0],
                        [-np.sin(euler[1]), 0, np.cos(euler[1])]
                        ])
        R_z = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                        [np.sin(euler[2]), np.cos(euler[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    
    def euler2phi(self, euler):
        R = self.euler2rotm(euler)
        # Get theta and phi from rotation matrix. theta is angle, phi is rotation axis
        theta = math.acos((R[0,0]+R[1,1]+R[2,2]-1)/2)
        phi = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*math.sin(theta))
        return theta, phi
    
    def rotm2phi(self, R):
        theta = math.acos((R[0,0]+R[1,1]+R[2,2]-1)/2)
        phi = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*math.sin(theta))
        return theta, phi   # norm phi should be 1
    
    def rotm2euler(self, R):
        sy = math.sqrt(R[2,2] * R[2,2] +  R[2,1] * R[2,1])
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
    
        return np.array([x, y, z])
    
    def phi2rotm(self, phi, theta):    # Rodrigues' rotation formula
        phi_hat = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
        R = np.eye(3) + phi_hat*math.sin(theta) + np.dot(phi_hat, phi_hat) * (1 - math.cos(theta))
        return R
    
    def get_robo_omega(self, t_K, theta_0, theta_dot_0, theta_f, theta_dot_f):
        # get the robot attitude (omega and angular velocity) at time t, with 4-order fitting
        sigma = math.sqrt(self.w1/self.w2)
        t0 = self.t0
        tf = self.tf
        phi_m0 = theta_0
        dphi_m0 = theta_dot_0
        phi_cf = theta_f
        dphi_cf = theta_dot_f
        d = 4 + sigma*(tf-t0)*(math.exp(sigma*tf)-math.exp(-sigma*tf)) - 2*(math.exp(sigma*tf)+math.exp(-sigma*tf))
        k2 = (((math.exp(-sigma*(tf-t0))-1)/sigma+tf-t0) * dphi_cf + (math.exp(-sigma*(tf-t0))-1)*(phi_cf-phi_m0) - ((math.exp(-sigma*(tf-t0))-1)/sigma+math.exp(-sigma*(tf-t0))*(tf-t0)) * dphi_m0)/d
        k3 = ((-(math.exp(sigma*(tf-t0))-1)/sigma+tf-t0) * dphi_cf + (math.exp(sigma*(tf-t0))-1)*(phi_cf-phi_m0) + ((math.exp(sigma*(tf-t0))-1)/sigma-math.exp(sigma*(tf-t0))*(tf-t0)) * dphi_m0)/d
        k1 = dphi_m0 - sigma*k2 + sigma*k3
        k0 = phi_m0 - k2 - k3

        # 机械臂末端在拦截时刻tf的位置和速度
        phi_mK = k0 + k1*(t_K-t0) + k2*math.exp(sigma*(t_K-t0)) + k3*math.exp(-sigma*(t_K-t0))
        dphi_mK = k1 + sigma*k2*math.exp(sigma*(t_K-t0)) - sigma*k3*math.exp(-sigma*(t_K-t0))
        
        return phi_mK, dphi_mK
        
    def pline(self):
        # get the rotation matrix R_process from initial attitude to final attitude
        R_0 = self.euler2rotm(self.robo_omega0)
        R_f = self.euler2rotm(self.robo_omegaf)
        R_process = np.matmul(R_0.T, R_f)
        
        # get the rotation axis and angle from R_process, represent as phi and theta
        theta_f, phi = self.rotm2phi(R_process)
        
        theta_0 = 0 # whatever the robot_omega0 is, theta_0 is the rotation start point w.r.t theta_f
        theta_dot_0 = 0 # robot is static at the beginning
        theta_dot_f = np.linalg.norm(self.robo_wf) # APPROXIMATED by the norm of the final angular velocity
        
        euler_k_all = np.array([])
        
        for t_k in np.arange(self.t0, self.tf, self.tf/1000):   # here 1000 match the delta time in approach_pos.py line 80
            theta_k, theta_dot_k = self.get_robo_omega(t_k, theta_0, theta_dot_0, theta_f, theta_dot_f)
            R_k = self.phi2rotm(phi, theta_k)   # R_k this line ISNNOT w.r.t to the world frame, BUT w.r.t the initial frame(R_0)
            R_k = R_k @ R_0     # ???????????????????????????
            euler_k = self.rotm2euler(R_k)
            euler_k_all = np.append(euler_k_all, euler_k)
            
        # robot orientation at each step
        euler_k_all = np.reshape(euler_k_all, (-1, 3))
        # object orientation at each step
        omega_all = self.predict()[4]
        
        return euler_k_all, omega_all
            
            
    
if __name__ == '__main__':
    
    # Stage 1: object parabola
    w0 = np.array([0.1, -0.5, -0.2])
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tau_max = np.array([1, 1, 1])
    omega0 = np.array([0.2, 1.5, 0])
    d_t = 1e-2
    
    x0=100
    y0=100
    z0=100
    v0=np.array([0.2, 0.5, 0.3])
    a=np.array([0, 0, -9.81])
    d_time = 1.15
    
    # Stage 2: robot approaches to object
    w1 = 0.5
    w2 = 0.5
    t0 = 0
    robo_omega0 = np.array([0.5, -0.75, -1])   
    
    robot_attitude = robot_attitude(w0=w0, I=I, tau_max=tau_max, omega0=omega0, d_t=d_t, x0=x0, y0=y0, z0=z0, v0=v0, a=a, d_time = d_time,
                                    w1=w1, w2=w2, t0=t0, robo_omega0=robo_omega0)
    euler_k_all, omega_all = robot_attitude.pline()
    
    print('error is:' + str(np.linalg.norm(euler_k_all[-1]-omega_all[-1])))
    
    # Static plot
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # roll_r = euler_k_all[:,0]
    # pitch_r = euler_k_all[:,1]
    # yaw_r = euler_k_all[:,2]
    # roll_o = omega_all[:,0]
    # pitch_o = omega_all[:,1]
    # yaw_o = omega_all[:,2]
    
    # ax.plot(roll_r, pitch_r, yaw_r, 'r', label='robot')
    # ax.plot(roll_o, pitch_o, yaw_o, 'b', label='object')
    # ax.set_xlabel('roll')
    # ax.set_ylabel('pitch')
    # ax.set_zlabel('yaw')
    # ax.set_title('Euler angle error is:' + str(np.linalg.norm(euler_k_all[-1]-omega_all[-1])))
    # ax.scatter(roll_r[0], pitch_r[0], yaw_r[0], c='r', marker='o', label='robot start')
    # ax.scatter(roll_r[-1], pitch_r[-1], yaw_r[-1], c='r', marker='x', label='robot end')
    # ax.scatter(roll_o[0], pitch_o[0], yaw_o[0], c='b', marker='o', label='object start')
    # ax.scatter(roll_o[-1], pitch_o[-1], yaw_o[-1], c='b', marker='x', label='object end')
    # ax.legend()
    # plt.show()
    
    # Dynamic plot
    def update(frame):
        plot1.set_data(euler_k_all[:frame,0], euler_k_all[:frame,1])
        plot1.set_3d_properties(euler_k_all[:frame,2])
        
        plot2.set_data(omega_all[:frame,0], omega_all[:frame,1])
        plot2.set_3d_properties(omega_all[:frame,2])
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    
    plot1 = ax1.plot([], [], [], 'r', label='robot')[0]
    plot2 = ax1.plot([], [], [], 'b', label='object')[0]
    
    x_min = min(np.min(euler_k_all[:,0]), np.min(omega_all[:,0]))
    x_max = max(np.max(euler_k_all[:,0]), np.max(omega_all[:,0]))
    y_min = min(np.min(euler_k_all[:,1]), np.min(omega_all[:,1]))
    y_max = max(np.max(euler_k_all[:,1]), np.max(omega_all[:,1]))
    z_min = min(np.min(euler_k_all[:,2]), np.min(omega_all[:,2]))
    z_max = max(np.max(euler_k_all[:,2]), np.max(omega_all[:,2]))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)
    ax1.set_xlabel('roll')
    ax1.set_ylabel('pitch')
    ax1.set_zlabel('yaw')
    ax1.set_title('Approach Altitude')
    ax1.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=len(euler_k_all), interval=10, repeat=False, blit=False)
    ani.save('approach_alt.gif', writer='pillow')