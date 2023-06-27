# Transform the center point to the contact point
# r_c = R * r_oc + p_o
# %%
__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np
import math 
from matplotlib import pyplot as plt
from adaptation import adapt
from forward_dynamics import forward_dynamics
from tqdm import tqdm
import time

class transform():
    def __init__(self, p_rk_all,  q_rk_all, v_rk_all, w_rk_all, contact_position):
        self.center_position = p_rk_all         # center position
        self.center_orientation = q_rk_all      # center orientation
        self.center_velocity = v_rk_all         # center velocity
        self.center_angular_velocity = w_rk_all # center angular velocity
        self.contact_position = contact_position        # contact position w.r.t center
    
    def rotation_matrix_from_quat(self):
        # get the rotation matrix from unit quaternion
        w = self.center_orientation[:, 0]
        x = self.center_orientation[:, 1]
        y = self.center_orientation[:, 2]
        z = self.center_orientation[:, 3]
        R = np.array([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
                        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
                        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]])
        length = self.center_orientation.shape[0]
        return R, length
    
    def center2contact_p(self):
        # get the rotation matrix from unit quaternion
        R, length = self.rotation_matrix_from_quat()
        
        # get the translation vector
        p = self.center_position
        
        c = np.zeros((length, 3))
        # transform
        for i in tqdm(range(length)):
            c[i] = np.matmul(R[:,:,i], self.contact_position) + p[i,:]
            
        return c
    
    def center2contact_v(self):
        R, length = self.rotation_matrix_from_quat()
        v_contact = np.zeros((length, 3))
        for i in tqdm(range(length)):
            v_contact[i] = np.cross(self.center_angular_velocity[i, :] , np.matmul(R[:,:,i], self.contact_position)) + self.center_velocity[i, :]
            
        return v_contact
    
    def trans(self):
        contact_p_rk_all = self.center2contact_p()
        contact_q_rk_all = self.center_orientation
        contact_w_rk_all = self.center_angular_velocity
        contact_v_rk_all = self.center2contact_v()
        
        return contact_p_rk_all, contact_q_rk_all, contact_v_rk_all, contact_w_rk_all
    
# %%
# if __name__ == '__main__':
begin_time = time.process_time()

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
N = 200                                 # number of steps
delta = 0.01                            # to stop when the velocity is close enough to the final velocity
g = 9.81

# Initial and final conditions
v_0 = np.array([2, 3, -2])
v_f = np.array([-0.5, -1, 1.5])
w_0 = np.array([2, -1, 1])
w_f = np.array([-1, 1, 3])

p_0 = np.array([0, 0, 0])
q_0 = np.array([1, 0, 0, 0])    # [w, x, y, z]

# contact point parameters (initial status is vertical to the ground)
# ----
# |  |          ^: z
# |  |          ->: y
# |  |c         /: x
# |  |
# |  |          c: contact point
# | o|          o: center point
# |  |  
# |  |
# |  |          length: 0.4m
# |  |          width: 0.1m
# |  |          c = o + [0, 0.05, 0.05*tan(pi/3)]
# ----

adapt_obj = adapt(v_0, w_0, m, I, f_max_vec, tau_max_vec, v_f, w_f, N, delta)
t_f, k, f_star_all, tau_star_all, v_all, w_all = adapt_obj.pline()

forward_dynamics_obj = forward_dynamics(t_f, f_star_all, tau_star_all, p_0, q_0, v_0, w_0, m, I, g, k)
y, p_rk_all, q_rk_all, v_rk_all, w_rk_all, euler_rk_all = forward_dynamics_obj.runge_kutta_method()

contact_position_body = np.array([0, 0.05, 0.05*math.tan(math.pi/3)])

trans2contact = transform(p_rk_all, q_rk_all, v_rk_all, w_rk_all, contact_position_body)
contact_position_spatial, contact_quaternion_spatial, contact_vel_spatial, contact_w_spatial = trans2contact.trans()

print("The total time is: ", time.process_time() - begin_time, "s")

temp = contact_position_spatial - p_rk_all
fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(temp[:,0], temp[:,1], temp[:,2], color='r', label='distance vector between center point and contact point')
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2 = fig.add_subplot(222, projection='3d')
ax2.plot(contact_position_spatial[:,0], contact_position_spatial[:,1], contact_position_spatial[:,2], color='b', label='contact point')
ax2.plot(p_rk_all[:,0], p_rk_all[:,1], p_rk_all[:,2], color='r', label='center point')
ax2.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
print('The distance between the center point and the contact point is: \n', GREEN + str(np.linalg.norm(temp, axis=1)) + RESET)

ax3 = fig.add_subplot(223, projection='3d')
ax3.plot(contact_vel_spatial[:,0], contact_vel_spatial[:,1], contact_vel_spatial[:,2], color='b', label='contact velocity')
ax3.plot(v_rk_all[:,0], v_rk_all[:,1], v_rk_all[:,2], color='r', label='center velocity')
ax3.legend()
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

temp_v = contact_vel_spatial - v_rk_all
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(temp_v[:,0], temp_v[:,1], temp_v[:,2], color='r', label='velocity vector between center point and contact point')
ax4.legend()
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')
print('The distance between the center point and the contact point is: \n', GREEN + str(np.linalg.norm(temp_v, axis=1)) + RESET)

plt.show()
# %%
