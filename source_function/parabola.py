__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
from mpmath import norm, cos, sin
from sympy import *
from sympy import Matrix
import math
from matplotlib import pyplot as plt

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

# 输入：物体t0时刻的position/orientation/angular velocity，任意给定时刻tf_K
# 输出：物体tf_K时刻的角度,姿态
class parabola:

    def __init__(self, w0, I, tau_max, omega0, d_t, x0, y0, z0, v0, a, d_time):
        self.w0 = w0
        self.I = I
        self.tau_max = tau_max
        self.omega0 = omega0
        self.d_t = d_t          # delta time
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.v0 = v0
        self.a = a
        self.d_time = d_time    # final time
        
    # % 输入：物体t时刻的姿态(角度omega0和角速度w0）
    # % 输出：物体t+d_t时刻的角度。
    def get_omegaf_k(self, omega0, w0):
        R0 = self.euler2rotm(omega0)
        R0_next = np.dot(self.aw(w0), R0)    #w_0是初始角速度, aw function includes d_t, so no d_t here.
        R0 = R0_next
        omegaf_k = self.rotm2euler(R0)
        return omegaf_k

    # 旋转矩阵转换为欧拉角：roll, pitch, yaw
    def rotm2euler(self, R):
    
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
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


    # Aw( w0 )将angular velocity 转换为3x3的旋转矩阵
    def aw(self, w0):
        if norm(w0) == 0:
            return np.eye(3)
        else:
            th = norm(w0) * self.d_t # d_t为时间间隔，这里取1，th是旋转的角度的模
            w = w0 / norm(w0)   # 单位化
            skew_w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            E_0 = np.eye(3) + skew_w * sin(th) + np.dot(skew_w, skew_w) * (1 - cos(th)) #Rodrigues公式
            E_0 = np.array(E_0, dtype=np.float64)
            return E_0

    def traPrediction(self, p_start, t, v0, a):
        v0_x = v0[0]
        v0_y = v0[1]
        v0_z = v0[2]
        a_x = a[0]
        a_y = a[1]
        a_z = a[2]
        x = p_start[0] + v0_x * t + 0.5 * a_x * t ** 2
        y = p_start[1] + v0_y * t + 0.5 * a_y * t ** 2
        z = p_start[2] + v0_z * t + 0.5 * a_z * t ** 2
        v_x = v0_x + a_x * t
        v_y = v0_y + a_y * t
        v_z = v0_z + a_z * t

        return np.array([x, y, z, v_x, v_y, v_z])

    def predict(self):
        w_next = self.w0
        w = self.w0
        omegaf_k = self.omega0
        omege_all = np.array(self.omega0)

        for t in np.arange(self.d_t, self.d_time, self.d_t): 
            omegaf_k = self.get_omegaf_k(omegaf_k, w)    
            omege_all = np.append(omege_all, omegaf_k) 
            x, y, z, v_x, v_y, v_z= self.traPrediction([self.x0, self.y0, self.z0], t, self.v0, self.a)

            # print('时间', MAGENTA + str(np.around(t,4)) + RESET, '|', '  预测的位置：', GREEN + str(np.around([x, y, z], 3)) + RESET, 
            #     ' | ', '  预测的角度：', RED + str(np.around(omegaf_k, 3)) + RESET)
            
        omege_all = np.reshape(omege_all, (-1, 3))
        return [x, y, z], omegaf_k, [v_x, v_y, v_z], self.w0, omege_all  # return the final position, euler angle and angular velocity


if __name__ == '__main__':
    
    w0 = np.array([2, 0.2, 0.2])
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tau_max = np.array([1, 1, 1])
    omega0 = np.array([0, 0, 0])
    d_t = 1e-2
    
    x0=100
    y0=100
    z0=100
    v0=np.array([0.2, 0.5, 0.3])
    a=np.array([0, 0, -9.81])
    d_time = 2.5

    obj = parabola(w0=w0, I=I, tau_max=tau_max, omega0=omega0, d_t=d_t, x0=x0, y0=y0, z0=z0, v0=v0, a=a, d_time = d_time)
    pos_f, euler_f, v_f, w_f, omega_all = obj.predict()