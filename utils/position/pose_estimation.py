import numpy as np
import cv2
import time
import itertools
from scipy import signal

class PoseEstimation():
    def __init__(self, wight, heigh):
        
        self.W = wight
        self.H = heigh
        self.X, self.Y, self.Z = [], [], []
    
    @staticmethod
    def Vel(deltaX, deltaT):
        return deltaX/deltaT
    
    @staticmethod
    def Knn(cy, cx, depth_image):
        a = np.linspace(cy-3, cy+3,7,dtype=int)
        b = np.linspace(cx-3, cx+3,7, dtype=int)
        c = list(itertools.product(a, b))
        ## All knn
        knn = np.array([depth_image[i[0], i[1]] for i in c])

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
            
            D_knn = self.Knn(cy, cx, depth_image)
            if D_knn != None: coordZ.append(D_knn)
            
        self.X.append(coordX)
        self.Y.append(coordY)
        self.Z.append(signal.medfilt(coordZ , 5))
        
        endX = time.time()
        try:
            velX = self.Vel(np.array(self.X[-1])-np.array(self.X[-2]), endX-startX)
        except:
            velX = self.Vel(np.zeros(21), endX-startX)
            
        endY = time.time()    
        try:
            velY = self.Vel(np.array(self.Y[-1])-np.array(self.Y[-2]), endY-startY)
        except:
            velY = self.Vel(np.zeros(21), endY-startY)
            
        endZ = time.time()    
        try:
            velZ = self.Vel(np.array(self.Z[-1])-np.array(self.Z[-2]), endZ-startZ)
        except:
            velZ = self.Vel(np.zeros(21), endZ-startZ)
        
        if np.all(velX > 0) or np.all(velX < 0):
            if np.all(np.array(self.X[-1])-np.array(self.X[-2]) > 5):
                print("Left")
            elif np.all(np.array(self.X[-1])-np.array(self.X[-2]) < -5):
                print("Right")
                
                
        if np.all(velY > 0) or np.all(velY < 0):
            if np.all(np.array(self.Y[-1])-np.array(self.Y[-2]) > 5):
                print("Down")
            elif np.all(np.array(self.Y[-1])-np.array(self.Y[-2]) < -5):
                print("Up")
                
                
        if np.all(velZ > 0) or np.all(velZ < 0):
            if np.all(np.array(self.Z[-1])-np.array(self.Z[-2]) > 10):
                print("Back")
            elif np.all(np.array(self.Z[-1])-np.array(self.Z[-2]) < -10):
                print("Forward")
            
        
        
    
    