import numpy as np
import cv2
import time
import itertools
from scipy import signal
from collections import deque
import pyrealsense2 as rs

class PoseEstimation():
    def __init__(self, wight, heigh):
        
        self.W = wight
        self.H = heigh
        self.X, self.Y, self.Z = deque(maxlen=2), deque(maxlen=2), deque(maxlen=2)
        
        self.X_res = 0
        self.Y_res = 0
        self.Z_res = 0
    
    @staticmethod
    def Vel(deltaX, deltaT):
        return deltaX/deltaT
    
    @staticmethod
    def Knn(cy, cx, depth_image):
        
        depth_intrin = depth_image.profile.as_video_stream_profile().intrinsics ############
        
        a = np.linspace(cy-3, cy+3,7,dtype=int)
        b = np.linspace(cx-3, cx+3,7, dtype=int)
        c = list(itertools.product(a, b))
        ## All knn
        knn = np.array([depth_image.get_distance(i[1], i[0]) for i in c])
        # knn = np.array([depth_image[i[0], i[1]] for i in c])

        ## Knn w/o zeros
        knn = knn[knn != 0]

        ## Knn w/o noise
        knn.sort()
        diff = np.diff(knn)

        try:
            ind = np.argmax(diff)
            if abs(diff[ind]) > 50:
                knn = knn[:ind+1]
        except:
            return None

        return np.mean(knn)
        
    
    def PoseStart(self, handLms, depth_image):
        startX = time.time()
        startY = time.time()
        startZ = time.time()
        coordX, coordY, coordZ = [], [], []
        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x*self.W), int(lm.y*self.H)
            coordX.append(cx)
            coordY.append(cy)

            # coordX.append(cx)
            # coordY.append(cy)

            
            D_knn = self.Knn(cy, cx, depth_image)
            # D_knn = self.Knn(cy, cx, depth_image)
            if D_knn != None: coordZ.append(D_knn)
        
        self.X.append(coordX)
        self.Y.append(coordY)
        self.Z.append(signal.medfilt(coordZ , 5))
        
        
        endX = time.time()
        try:
            velX = self.Vel(np.array(self.X[-1])-np.array(self.X[-2]), endX-startX)
        except:
            velX = self.Vel(np.zeros(21), endX-startX)
            self.X_res = np.mean(np.array(self.X[0]))
            
        endY = time.time()    
        try:
            velY = self.Vel(np.array(self.Y[-1])-np.array(self.Y[-2]), endY-startY)
        except:
            velY = self.Vel(np.zeros(21), endY-startY)
            self.Y_res = np.mean(np.array(self.Y[0]))
            
        endZ = time.time()    
        try:
            velZ = self.Vel(np.array(self.Z[-1])-np.array(self.Z[-2]), endZ-startZ)
        except:
            velZ = self.Vel(np.zeros(21), endZ-startZ)
            
        self.X_res = np.mean(np.array(self.X[0]))
        self.Y_res = np.mean(np.array(self.Y[0]))
        self.Z_res = np.mean(np.array(self.Z[0]))
            
        if np.all(velX > 0): 
            self.X_res = np.mean(np.array(self.X[1]))
            # self.X_res += np.mean(np.array(self.X[0]) - np.array(self.X[1]))
            # print("Left")
        elif np.all(velX < 0):
            self.X_res = np.mean(np.array(self.X[1]))
            # self.X_res += np.mean(np.array(self.X[0]) - np.array(self.X[1]))
        # print(self.X_res)
            
        if np.all(velY > 0):
            self.Y_res = np.mean(np.array(self.Y[1]))
            # self.Y_res += np.mean(np.array(self.Y[0]) - np.array(self.Y[1]))
            # print(self.Y_res)
        elif np.all(velY < 0):
            self.Y_res = np.mean(np.array(self.Y[1]))
            # self.Y_res += np.mean(np.array(self.Y[0]) - np.array(self.Y[1]))
            # print(self.Y_res)
            
#         if np.all(velZ > 0):
#             self.Z_res = np.mean(np.array(self.Z[1]))
#             # self.Z_res += np.mean(np.array(self.Z[0])) - np.mean(np.array(self.Z[1]))
#             # print("Back")
#         elif np.all(velZ < 0):
#             self.Z_res = np.mean(np.array(self.Z[1]))
#             # self.Z_res += np.mean(np.array(self.Z[0])) - np.mean(np.array(self.Z[1]))
            
        # depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(self.Y_res), int(self.X_res)], self.Z_res) 
        depth_intrin = depth_image.profile.as_video_stream_profile().intrinsics
        depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(self.Y_res), int(self.X_res)], self.Z_res)

            
        # return self.X_res, self.Y_res, self.Z_res
        return depth_point_in_meters_camera_coords[1], depth_point_in_meters_camera_coords[0], depth_point_in_meters_camera_coords[2]
            # print("Right")
        
#         if np.all(velX > 0) or np.all(velX < 0):
#             if np.all(np.array(self.X[-1])-np.array(self.X[-2]) > 5):
#                 print("Left")
#             elif np.all(np.array(self.X[-1])-np.array(self.X[-2]) < -5):
#                 print("Right")
                
                
#         if np.all(velY > 0) or np.all(velY < 0):
#             if np.all(np.array(self.Y[-1])-np.array(self.Y[-2]) > 5):
#                 print("Down")
#             elif np.all(np.array(self.Y[-1])-np.array(self.Y[-2]) < -5):
#                 print("Up")
                
                
#         if np.all(velZ > 0) or np.all(velZ < 0):
#             if np.all(np.array(self.Z[-1])-np.array(self.Z[-2]) > 10):
#                 print("Back")
#             elif np.all(np.array(self.Z[-1])-np.array(self.Z[-2]) < -10):
#                 print("Forward")
            
        
        
    
    