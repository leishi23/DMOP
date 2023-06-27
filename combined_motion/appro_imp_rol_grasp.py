# %%
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
from source_motion.approach_alt import robot_attitude
from source_motion.approach_pos import robot_position
from source_motion.impact import impact
from source_motion.rolling import rolling
from source_motion.grasp_fixed import grasp_fixed
from source_motion.grasp_semifixed import adapt_semi
from source_function.forward_dynamics import forward_dynamics
from source_function.parabola import parabola
from matplotlib import pyplot as plt
import json

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'


# %% Stage 1-1: Approach-pos
g = 9.81

# object initial position and velocity
x0, y0, z0 = 1, 0, 3
vx, vy, vz = -5, -2, 1
# robot initial position and velocity
x1, y1, z1 = 1, 1, 0.5
vx1, vy1, vz1 = 5, 0, 0

t0, tf = 0, 0.5
w1 = 0.5    # weight
w2 = 0.5

rm_t0 = np.array([x1, y1, z1])  # robot initial position
drm_t0 = np.array([vx1, vy1, vz1])  # robot initial velocity
rc_t0 = np.array([x0, y0, z0])  # object initial position
drc_t0 = np.array([vx, vy, vz])  # object initial velocity
ddrc_t0 = np.array([0, 0, -g])  # object acceleration

robot_position = robot_position(w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf)
p = robot_position.pline3d(w1, w2, rm_t0, drm_t0, rc_t0, drc_t0, ddrc_t0, t0, tf)
rob_pos = np.matrix(p[-3:]).tolist()
obj_pos = np.matrix(p[:3]).tolist()
rob_pos_aft_appro = np.array([p[-3][-1], p[-2][-1], p[-1][-1]])
obj_pos_aft_appro = np.array([p[0][-1], p[1][-1], p[2][-1]])

fig = plt.figure(figsize=(10, 10))

ax_1_1 = fig.add_subplot(111, projection='3d')
x = np.array(p[0])
y = np.array(p[1])
z = np.array(p[2])
rm_x = np.array(p[3])
rm_y = np.array(p[4])
rm_z = np.array(p[5])
ax_1_1.plot(x, y, z, 'blue', label='Object')
ax_1_1.plot(rm_x, rm_y, rm_z, 'red', label='Robot')
ax_1_1.legend()
ax_1_1.set_xlabel('X')
ax_1_1.set_ylabel('Y')
ax_1_1.set_zlabel('Z')
ax_1_1.set_title('Approach Position')
# plt.show()


# %% Stage 1-2: Approach-alt
# Stage 1-2-1: object parabola
# Object initial angular velocity
w0 = np.array([5, 3, 10])    # TBD

# Object Inertial and max torque
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
tau_max = np.array([10, 10, 10])

# Object initial orientation
omega0 = np.array([0, 0, 0])

# Object initial velocity and acceleration
v0=np.array([vx, vy, vz])
a=np.array([0, 0, -9.81])

# Final time
d_time = tf

# Dealta time in parabola predictiion
d_t = d_time/1000   # here 1000 match the apprach_pos.py line 80

# Stage 1-2-2: robot approaches to object
t0 = 0
# Robot initial orientation
robo_omega0 = np.array([0, 0, 0])  

robot_attitude = robot_attitude(w0=w0, I=I, tau_max=tau_max, omega0=omega0, d_t=d_t, x0=x0, y0=y0, z0=z0, v0=v0, a=a, d_time = d_time,
                                w1=w1, w2=w2, t0=t0, robo_omega0=robo_omega0)
euler_k_all, omega_all = robot_attitude.pline() # robot attitude, object attitude
rob_alt = np.matrix(euler_k_all.T).tolist()
obj_alt = np.matrix(omega_all.T).tolist()
rob_omega_aft_appro = euler_k_all[-1]
obj_omega_aft_appro = omega_all[-1]

fig = plt.figure(figsize=(10, 10))
ax_1_2 = fig.add_subplot(111, projection='3d')
roll_r = euler_k_all[:,0]
pitch_r = euler_k_all[:,1]
yaw_r = euler_k_all[:,2]
roll_o = omega_all[:,0]
pitch_o = omega_all[:,1]
yaw_o = omega_all[:,2]

ax_1_2.plot(roll_r, pitch_r, yaw_r, 'r', label='robot')
ax_1_2.plot(roll_o, pitch_o, yaw_o, 'b', label='object')
ax_1_2.set_xlabel('roll')
ax_1_2.set_ylabel('pitch')
ax_1_2.set_zlabel('yaw')
ax_1_2.set_title('Approach Altitude Euler angle error is:' + str(np.linalg.norm(euler_k_all[-1]-omega_all[-1])))
ax_1_2.scatter(roll_r[0], pitch_r[0], yaw_r[0], c='r', marker='o', label='robot start')
ax_1_2.scatter(roll_r[-1], pitch_r[-1], yaw_r[-1], c='r', marker='x', label='robot end')
ax_1_2.scatter(roll_o[0], pitch_o[0], yaw_o[0], c='b', marker='o', label='object start')
ax_1_2.scatter(roll_o[-1], pitch_o[-1], yaw_o[-1], c='b', marker='x', label='object end')
ax_1_2.legend()
# plt.show()

## Dump approach data into local file
appro_file_path = os.path.join(os.getcwd(), 'data/appro_data')
data_dict = {"position":{"robot position": rob_pos, "object position":obj_pos}, "altitude":{"robot altitude": rob_alt, "object altitude": obj_alt}}
json_object_appro = json.dumps(data_dict, indent=4)
with open(appro_file_path, "w") as outfile:outfile.write(json_object_appro)
# %% Stage 2: Impact 
obj_init_pos = obj_pos_aft_appro
obj_init_vel = np.array([0, 0, 0])
obj_init_ang = obj_omega_aft_appro
obj_init_ang_vel = w0

# Set the object post velocity and angular velocity, and enable them to the object
obj_post_vel = np.array([0.75, 1, 0.5])
obj_post_ang_vel = np.array([1, 0.25, 0.5])

impact = impact(obj_init_pos, obj_init_vel, obj_init_ang, obj_init_ang_vel, obj_post_vel, obj_post_ang_vel)
state = impact.obj_impact() 
obj_post_imp_pos = state[0]
obj_post_imp_vel = state[1]
obj_post_imp_omega = state[2]
obj_post_imp_ang_vel = state[3]

print("Post position is: " + str(state[0]))
print("Post velocity is: " + str(state[1]))
print("Post orientation is: " + str(state[2]))
print("Post angular velocity is: " + str(state[3]))


# %% Stage 3: Rolling

# Only consider the one axis rotation
m = 1

v_0_3 = obj_post_imp_vel
w_0_3 = obj_post_imp_ang_vel
w_max_3 = np.array([math.pi/2, 0, 0])       # constrained by max angular velocity
tau_max_3 = np.array([tau_max[0], 0, 0])

omega_0_3 = np.array([obj_post_imp_omega[0], 0 ,0])
omega_f_3 = np.array([math.pi/2, 0, 0])     # rotate around x axis  

# Convert the center params to contact params
def center2contact(center_pos, contact_pos_bodyframe, euler):
    # convert omega into rotation matrix
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
    contact_pos = np.matmul(R, contact_pos_bodyframe) + center_pos
    return contact_pos
    
p_0_3 = center2contact(obj_post_imp_pos, np.array([0, -0.1, -0.05]), omega_0_3)
l_c = 0.1

# contact point parameters (initial status is parallel to the ground)
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

roll = rolling(m, I, tau_max_3, v_0_3, w_0_3, w_max_3, omega_0_3, omega_f_3, p_0_3, l_c)
p_all, v_all, w_all, omega_all = roll.pline()
p_post_roll = p_all[-1]
v_post_roll = v_all[-1]
w_post_roll = w_all[-1]
omega_post_roll = omega_all[-1]

# plot the trajectory
fig = plt.figure(figsize=(15, 15))

ax31 = fig.add_subplot(221, projection='3d')
ax31.plot(p_all[:, 0], p_all[:, 1], p_all[:, 2], label='Position')
ax31.set_xlabel('x')
ax31.set_ylabel('y')
ax31.set_zlabel('z')
ax31.set_title('Rolling Position Trajcetory')
ax31.legend()

ax32 = fig.add_subplot(222, projection='3d')
ax32.plot(omega_all[:, 0], omega_all[:, 1], omega_all[:, 2], label='Orientation')
ax32.set_xlabel('x')
ax32.set_ylabel('y')
ax32.set_zlabel('z')
ax32.set_title('Rolling Orientation Trajcetory')
ax32.legend()

ax33 = fig.add_subplot(223, projection='3d')
ax33.plot(v_all[:, 0], v_all[:, 1], v_all[:, 2], label='Linear Velocity')
ax33.set_xlabel('x')
ax33.set_ylabel('y')
ax33.set_zlabel('z')
ax33.set_title('Rolling Linear Velocity Trajcetory')
ax33.legend()

ax34 = fig.add_subplot(224)
ax34.plot(w_all[:, 0], label='Angular Velocity')
ax34.set_xlabel('time')
ax34.set_ylabel('Angular Velocity')
ax34.set_title('Rolling Angular Velocity Trajcetory')
ax34.legend()


# %% Stage 4: Grasping

# When grasping, regard robot and object as the whole
# Object parameters
f_max_vec_4 = np.array([10, 10, 10])
tau_max_vec_4 = np.array([10, 10, 10])
N = 200         # number of steps to iterate through t_f
delta = 0.01    # to stop when the velocity is close enough to the final velocity

# Convert the center params to contact params
def contact2center(contact_pos, contact_pos_bodyframe, euler):
    # convert omega into rotation matrix
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
    center_pos = contact_pos - np.matmul(R, contact_pos_bodyframe) 
    return center_pos
    
p_0_4 = contact2center(p_post_roll, np.array([0, -0.1, -0.05]), omega_post_roll)

# Object initial and final conditions
v_0_4 = v_post_roll
v_f_4 = np.array([0, 0, 0])
w_0_4 = w_post_roll
w_f_4 = np.array([0, 0, 0])
# p_0_4 = p_post_roll
# Convert euler angle to unit quaternion
def euler_to_quaternion(rpy):
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

q_0_4 = euler_to_quaternion(omega_post_roll)    # [w, x, y, z]

# RK-4 to get the object trajectory
obj = grasp_fixed(v_0_4, w_0_4, m, I, f_max_vec_4, tau_max_vec_4, v_f_4, w_f_4, N, delta, p_0_4, q_0_4, g)
t_f, y, p_rk_all, q_rk_all, v_rk_all, w_rk_all, euler_rk_all, v_dot_rk_all, w_dot_rk_all = obj.res()

print(GREEN + "Final position:          " + RESET, CYAN + str(y[0]) + RESET)
print(GREEN + "Final quaternion:        " + RESET, CYAN + str(y[1]) + RESET)
print(GREEN + "Final velocity:          " + RESET, CYAN + str(y[2]) + RESET)
print(GREEN + "Final angular velocity:  " + RESET, CYAN + str(y[3]) + RESET)
print(GREEN + "Final time:              " + RESET, CYAN + str(t_f) + RESET)

fig = plt.figure(figsize=(15, 15))  
ax_41 = fig.add_subplot(3, 2, 1, projection='3d')
ax_41.plot(p_rk_all[:, 0], p_rk_all[:, 1], p_rk_all[:, 2], color='blue', label='Position trajectory')
ax_41.set_xlabel('x')
ax_41.set_ylabel('y')
ax_41.set_zlabel('z')
ax_41.set_title('Grasping Position Trajcetory')
ax_41.scatter(p_rk_all[0, 0], p_rk_all[0, 1], p_rk_all[0, 2], color='red', label='Initial position')
ax_41.scatter(p_rk_all[-1, 0], p_rk_all[-1, 1], p_rk_all[-1, 2], color='green', label='Final position')
ax_41.legend()

ax_42 = fig.add_subplot(3, 2, 2, projection='3d')
ax_42.plot(euler_rk_all[:, 0], euler_rk_all[:, 1], euler_rk_all[:, 2], color='blue', label='Euler angles trajectory')
ax_42.set_xlabel('roll')
ax_42.set_ylabel('pitch')
ax_42.set_zlabel('yaw')
ax_42.set_title('Grasping Euler Angles Trajcetory')
ax_42.scatter(euler_rk_all[0, 0], euler_rk_all[0, 1], euler_rk_all[0, 2], color='red', label='Initial Euler angles')
ax_42.scatter(euler_rk_all[-1, 0], euler_rk_all[-1, 1], euler_rk_all[-1, 2], color='green', label='Final Euler angles')
ax_42.legend()

ax_43 = fig.add_subplot(3, 2, 3, projection='3d')
ax_43.plot(v_rk_all[:, 0], v_rk_all[:, 1], v_rk_all[:, 2], color='blue', label='Linear velocity trajectory')
ax_43.set_xlabel('x')
ax_43.set_ylabel('y')
ax_43.set_zlabel('z')
ax_43.set_title('Grasping Linear Velocity Trajcetory')
ax_43.scatter(v_rk_all[0, 0], v_rk_all[0, 1], v_rk_all[0, 2], color='red', label='Initial velocity')
ax_43.scatter(v_rk_all[-1, 0], v_rk_all[-1, 1], v_rk_all[-1, 2], color='green', label='Final velocity')
ax_43.legend()

ax_44 = fig.add_subplot(3, 2, 4, projection='3d')
ax_44.plot(w_rk_all[:, 0], w_rk_all[:, 1], w_rk_all[:, 2], color='blue', label='Angular velocity trajectory')
ax_44.set_xlabel('x')
ax_44.set_ylabel('y')
ax_44.set_zlabel('z')
ax_44.set_title('Grasping Angular Velocity Trajcetory')
ax_44.scatter(w_rk_all[0, 0], w_rk_all[0, 1], w_rk_all[0, 2], color='red', label='Initial angular velocity')
ax_44.scatter(w_rk_all[-1, 0], w_rk_all[-1, 1], w_rk_all[-1, 2], color='green', label='Final angular velocity')
ax_44.legend()

ax_45 = fig.add_subplot(3, 2, 5, projection='3d')
ax_45.plot(v_dot_rk_all[:, 0], v_dot_rk_all[:, 1], v_dot_rk_all[:, 2], color='blue', label='Linear acceleration trajectory')
ax_45.set_xlabel('x')
ax_45.set_ylabel('y')
ax_45.set_zlabel('z')
ax_45.set_title('Grasping Linear Acceleration Trajcetory')
ax_45.scatter(v_dot_rk_all[0, 0], v_dot_rk_all[0, 1], v_dot_rk_all[0, 2], color='red', label='Initial acceleration')
ax_45.scatter(v_dot_rk_all[-1, 0], v_dot_rk_all[-1, 1], v_dot_rk_all[-1, 2], color='green', label='Final acceleration')
ax_45.legend()

ax_46 = fig.add_subplot(3, 2, 6, projection='3d')
ax_46.plot(w_dot_rk_all[:, 0], w_dot_rk_all[:, 1], w_dot_rk_all[:, 2], color='blue', label='Angular acceleration trajectory')
ax_46.set_xlabel('x')
ax_46.set_ylabel('y')
ax_46.set_zlabel('z')
ax_46.set_title('Grasping Angular Acceleration Trajcetory')
ax_46.scatter(w_dot_rk_all[0, 0], w_dot_rk_all[0, 1], w_dot_rk_all[0, 2], color='red', label='Initial angular acceleration')
ax_46.scatter(w_dot_rk_all[-1, 0], w_dot_rk_all[-1, 1], w_dot_rk_all[-1, 2], color='green', label='Final angular acceleration')
ax_46.legend()

plt.show() 
# %%
