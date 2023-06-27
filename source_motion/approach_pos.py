__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# Input: w1, w2, 机械臂初始位置rm_t0, 机械臂初始速度drm_t0, 物体初始位置rc_t0, 物体初始速度drc_t0, 物体初始加速度ddrc_t0, t0, 终止时间tf
# Output: x, y, z, rm_x, rm_y, rm_z 物体位置和机械臂位置
class robot_position:
    def __init__(self, w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf):
        self.t0 = t0  # 初始时间
        self.tf = tf  # 终止时间
        self.w1 = w1  # 权重1
        self.w2 = w2  # 权重2
        self.rc_t0 = rc_t0  # 初始位置
        self.drc_t0 = drc_t0  # 初始速度
        self.ddrc_t0 = ddrc_t0  # 初始加速度
        self.rm_t0 = rm_t0
        self.drm_t0 = drm_t0

    def sigma(self, w1, w2):
        sig = np.sqrt(w1 / w2)
        return sig

    def length(self, x0, v0, a, t):
        x = x0 + v0 * t + 0.5 * a * (t * t)
        return x

    def vec(self, v0, a, t):
        v = v0 + a * t
        return v

    def hamiton(self, w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0):
        t = tf - t0
        rc_tf = self.length(rc_t0, drc_t0, ddrc_t0, tf)
        drc_tf = self.vec(drc_t0, ddrc_t0, tf)
        h = rc_tf - rm_t0
        sig = self.sigma(w1, w2)
        d = 4 + sig * t * (math.exp(sig * tf) - math.exp(-sig * tf)) - 2 * (math.exp(sig * tf) + math.exp(-sig * tf)) 
        k2 = (((math.exp(-sig * t) - 1) / sig + t) * drc_tf + (math.exp(-sig * t) - 1) * h -
              ((math.exp(-sig * t) - 1) / sig + math.exp(-sig * t) * t) * drm_t0) / d
        k3 = ((-(math.exp(sig * (t)) - 1) / sig + t) * drc_tf + (math.exp(sig * (t)) - 1) * (h) +
              ((math.exp(sig * (t)) - 1) / sig - math.exp(sig * (t)) * (t)) * drm_t0) / d
        k1 = drm_t0 - sig * k2 + sig * k3
        k0 = rm_t0 - k2 - k3
        # H = 1 + w1*(k1.T*k1) - 4*(w1/w2)*(k2.T*k3) + 2*w1*sigma*((k2.T*math.exp(sigma*(tf-t0))-k3.T*math.exp(-sigma*(tf-t0)))* k1)- 2*w2*sigma**2*((k2.T*math.exp(sigma*(tf-t0))+k3.T*math.exp(-sigma*(tf-t0)))*ddrc_tf);
        return k0, k1, k2, k3

    def pline(self, w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf):
        x = np.array([])
        rm_x = np.array([])
        sig = self.sigma(w1, w2)
        k0 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[0]
        k1 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[1]
        k2 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[2]
        k3 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[3]
        for k in range(100):
            t = tf / 100 * k
            # x =np.append (x,x0 + vx*(t-t0))
            x = np.append(x, self.length(rc_t0, drc_t0, ddrc_t0, t))
            rm = k0 + k1 * (t - t0) + k2 * math.exp(sig * (t - t0)) + k3 * math.exp(-sig * (t - t0))
            rm_x = np.append(rm_x, rm)
        return x, rm_x

    def pline3d(self, w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        rm_x = np.array([])
        rm_y = np.array([])
        rm_z = np.array([])
        sig = self.sigma(w1, w2)
        k0 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[0]
        k1 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[1]
        k2 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[2]
        k3 = self.hamiton(w1, w2, t0, tf, rc_t0, drc_t0, ddrc_t0, rm_t0, drm_t0)[3]
        for k in range(1000):
            t = tf / 1000 * k
            # x =np.append (x,x0 + vx*(t-t0))
            x = np.append(x, self.length(rc_t0, drc_t0, ddrc_t0, t)[0])
            y = np.append(y, self.length(rc_t0, drc_t0, ddrc_t0, t)[1])
            z = np.append(z, self.length(rc_t0, drc_t0, ddrc_t0, t)[2])
            rm = k0 + k1 * (t - t0) + k2 * math.exp(sig * (t - t0)) + k3 * math.exp(-sig * (t - t0))
            rm_x = np.append(rm_x, rm[0])
            rm_y = np.append(rm_y, rm[1])
            rm_z = np.append(rm_z, rm[2])
        # print('x=', x, 'y=', y, 'z=', z, 'rm_x=', rm_x, 'rm_y=', rm_y, 'rm_z=', rm_z)
        return x, y, z, rm_x, rm_y, rm_z    # process position of object and robot

    def plot(self, w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z, rm_x, rm_y, rm_z = self.pline3d(w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf)
        ax.plot(x, y, z, label='line')
        ax.plot(rm_x, rm_y, rm_z, label='line')
        ax.legend()
        plt.show()


if __name__ == '__main__':

    g = 9.81

    # object initial position and velocity
    x0, y0, z0 = 0, 0, 5
    vx, vy, vz = 1, -1, 0.5
    # robot initial position and velocity
    x1, y1, z1 = 1, -1, 1
    vx1, vy1, vz1 = 0, 0, 0
    
    t0, tf = 0, 1.15
    w1 = 0.5    # weight
    w2 = 0.5

    rm_t0 = np.array([x1, y1, z1])  # robot initial position
    drm_t0 = np.array([vx1, vy1, vz1])  # robot initial velocity
    rc_t0 = np.array([x0, y0, z0])  # object initial position
    drc_t0 = np.array([vx, vy, vz])  # object initial velocity
    ddrc_t0 = np.array([0, 0, -g])  # object acceleration

    robot_position = robot_position(w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf)
    p = robot_position.pline3d(w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf)
    
    x = np.array(p[0])
    y = np.array(p[1])
    z = np.array(p[2])
    rm_x = np.array(p[3])
    rm_y = np.array(p[4])
    rm_z = np.array(p[5])

    # Static plot
    # fig = plt.figure(figsize=(15, 10))
    # ax_3d = fig.add_subplot(111, projection='3d')
    # ax_3d.plot(x, y, z, 'blue', label='Object')
    # ax_3d.plot(rm_x, rm_y, rm_z, 'red', label='Robot')
    # ax_3d.legend()
    # ax_3d.set_xlabel('X')
    # ax_3d.set_ylabel('Y')
    # ax_3d.set_zlabel('Z')
    # plt.show()
    
    # Dynamic plot
    def update(frame):
        plot1.set_data(x[:frame], y[:frame])
        plot1.set_3d_properties(z[:frame])
        
        plot2.set_data(rm_x[:frame], rm_y[:frame])
        plot2.set_3d_properties(rm_z[:frame])
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # ax.plot here returns a list of Line3D (len(list)=1), `[0]` here is to extract the first element from list
    plot1 = ax.plot([], [], [], 'blue', label='Object')[0]
    plot2 = ax.plot([], [], [], 'red', label='Robot')[0]
    
    x_min = min(min(x), min(rm_x))
    x_max = max(max(x), max(rm_x))
    y_min = min(min(y), min(rm_y))
    y_max = max(max(y), max(rm_y))
    z_min = min(min(z), min(rm_z))
    z_max = max(max(z), max(rm_z))
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Approach Position')
    ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=len(x), interval=5, repeat=False, blit=False)
    ani.save('approach_pos.gif', writer='pillow')