import urx
import numpy as np

class RobotInitPos():
    def __init__(self, ip):
        self.ip = ip
        
        rob = urx.Robot(self.ip)
        InitPosJointsAng = np.array([90, -90, 90, -90, 90, 0])/180*np.pi
        rob.movej(InitPosJointsAng, 0.3, 0.1)