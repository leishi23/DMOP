# 旋转矩阵转换为欧拉角：roll, pitch, yaw
import math
import numpy as np 
from tqdm import tqdm

# All Euler angles are in roll pitch yaw order

def rotm2euler(R):

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
def euler2rotm(euler):
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

def quat2rotm(quat):
    # get the rotation matrix from unit quaternion
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    R = np.array([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
                    [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
                    [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]])
    return R

# Convert euler angle to unit quaternion
def euler_to_quaternion(rpy):
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def rotm2quat(R):
    # get the unit quaternion from rotation matrix
    w = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    x = (R[2,1] - R[1,2]) / (4*w)
    y = (R[0,2] - R[2,0]) / (4*w)
    z = (R[1,0] - R[0,1]) / (4*w)
    return np.array([w, x, y, z])

# 旋转矩阵转换为轴角表示
def rotm2axisangle(R):
    theta = math.acos((np.trace(R) - 1) / 2)
    w = 1 / (2 * math.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w, theta

# 轴角表示转换为旋转矩阵
def axisangle2rotm(w, theta):
    w = w / np.linalg.norm(w)
    wx, wy, wz = w[0], w[1], w[2]
    w_hat = np.array([[0, -wz, wy],
                        [wz, 0, -wx],
                        [-wy, wx, 0]])
    R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * np.dot(w_hat, w_hat)
    return R

def center2contact_p(R, center_position, contact_position):
    # transform
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
    # contact_position_body = np.array([0, 0.05, 0.05*math.tan(math.pi/3)])
    c = np.matmul(R, contact_position) + center_position
    return c

# Convert the center params to contact params
def contact2center(contact_pos, contact_pos_bodyframe, euler):  
    # contact_pos_bodyframe is from center to contact
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

def center2contact_v(R, center_angular_velocity, center_velocity, contact_position):
    v_contact = np.cross(center_angular_velocity , np.matmul(R, contact_position)) + center_velocity
    return v_contact
