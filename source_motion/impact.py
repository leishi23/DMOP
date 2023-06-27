__author__ = "Lei Shi"
__copyright__ = "Copyright 2023, The Tencent Robotics-X Manipulation Project"
__email__ = "leishi9823@gmail.com"
__date__ = "2021/06/07"
import numpy as np

class impact():
    def __init__(self, obj_init_pos, obj_init_vel, obj_init_ang, obj_init_ang_vel, obj_post_vel, obj_post_ang_vel):
        self.obj_init_pos = obj_init_pos
        self.obj_init_vel = obj_init_vel
        self.obj_init_ang = obj_init_ang
        self.obj_init_ang_vel = obj_init_ang_vel
        self.obj_post_vel = obj_post_vel
        self.obj_post_ang_vel = obj_post_ang_vel
    
    def obj_impact(self):
        return np.array([self.obj_init_pos, self.obj_post_vel, self.obj_init_ang,  self.obj_post_ang_vel])
    
    
if __name__ == '__main__':
    obj_init_pos = np.array([0, 0, 0])
    obj_init_vel = np.array([0, 0, 0])
    obj_init_ang = np.array([0, 0, 0])
    obj_init_ang_vel = np.array([0, 0, 0])
    
    # Set the object post velocity and angular velocity, and enable them to the object
    obj_post_vel = np.array([3, 4, 1])
    obj_post_ang_vel = np.array([1, 2, 0.5])
    
    impact = impact(obj_init_pos, obj_init_vel, obj_init_ang, obj_init_ang_vel, obj_post_vel, obj_post_ang_vel)
    state = impact.obj_impact()
    print("Post position is: " + str(state[0]))
    print("Post velocity is: " + str(state[1]))
    print("Post angular is: " + str(state[2]))
    print("Post angular velocity is: " + str(state[3]))