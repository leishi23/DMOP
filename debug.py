import numpy as np

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
    R = R_z @ R_y @ R_x
    return R

p = np.array([0, 1, 0])
rpy = np.array([np.pi/3, np.pi/2, np.pi/6])
print(euler2rotm(rpy) @ p)

