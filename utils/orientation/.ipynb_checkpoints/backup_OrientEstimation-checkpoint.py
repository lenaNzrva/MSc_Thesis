import cv2
import numpy as np
from scipy.linalg import lstsq
import math
import time
from collections import deque

class OrientEstimation:
    def __init__(self, weight, height):
        
        self.W = weight
        self.H = height
        self.X = deque(maxlen=2)
        self.Y = deque(maxlen=2)
        # self.X_norm = deque(maxlen=2)
        # self.Y_norm = deque(maxlen=2)
        
        self.ang = 90
        # self.Len = deque(maxlen=2)
        
        self.startX = time.time()
        self.startY = time.time()
        
        
    @staticmethod
    def green_landmarks(self, coord_last):
        img = self.color_image
        th = 3
        color = (0,255,0)

        for i in range(4):
            cv2.line(img, (int(coord_last[i][1]), int(coord_last[i][2])), 
                     (int(coord_last[i+1][1]), int(coord_last[i+1][2])), color, th)

        for i in range(5, 8):
            cv2.line(img, (int(coord_last[i][1]), int(coord_last[i][2])), 
                     (int(coord_last[i+1][1]), int(coord_last[i+1][2])), color, th)

        for i in range(9, 12):
            cv2.line(img, (int(coord_last[i][1]), int(coord_last[i][2])), 
                     (int(coord_last[i+1][1]), int(coord_last[i+1][2])), color, th)

        for i in range(13, 16):
            cv2.line(img, (int(coord_last[i][1]), int(coord_last[i][2])), 
                     (int(coord_last[i+1][1]), int(coord_last[i+1][2])), color, th)

        for i in range(17, 20):
            cv2.line(img, (int(coord_last[i][1]), int(coord_last[i][2])), 
                     (int(coord_last[i+1][1]), int(coord_last[i+1][2])), color, th)


        img = cv2.line(img, (int(coord_last[0][1]), int(coord_last[0][2])), 
                       (int(coord_last[5][1]), int(coord_last[5][2])), color, th)
        img = cv2.line(img, (int(coord_last[0][1]), int(coord_last[0][2])), 
                       (int(coord_last[9][1]), int(coord_last[9][2])), color, th)
        img = cv2.line(img, (int(coord_last[0][1]), int(coord_last[0][2])), 
                       (int(coord_last[13][1]), int(coord_last[13][2])), color, th)
        img = cv2.line(img, (int(coord_last[0][1]), int(coord_last[0][2])), 
                       (int(coord_last[17][1]), int(coord_last[17][2])), color, th)

        img = cv2.line(img, (int(coord_last[5][1]), int(coord_last[5][2])), 
                       (int(coord_last[9][1]), int(coord_last[9][2])), color, th)
        img = cv2.line(img, (int(coord_last[9][1]), int(coord_last[9][2])), 
                       (int(coord_last[13][1]), int(coord_last[13][2])), color, th)
        img = cv2.line(img, (int(coord_last[13][1]), int(coord_last[13][2])), 
                       (int(coord_last[17][1]), int(coord_last[17][2])), color, 5)


        self.depth_image[np.sum(img == [0,255,0], axis=2) < 3] = 0

        return self.depth_image


    ## считаем угол плоскости
    @staticmethod
    def distance(self, a1, b1, c1, a2, b2, c2):
        d = ( a1 * a2 + b1 * b2 + c1 * c2 )
        e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
        e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
        d = d / (e1 * e2)
        return math.degrees(math.acos(d))
    
    @staticmethod
    def Vel(deltaX, deltaT):
        return deltaX/deltaT
    
    @staticmethod
    def Get_Yaw(self):
        coordX, coordY = [], []
        # coordX_norm,coordY_norm = [], []
        
        for id, lm in enumerate(self.handLms.landmark):
            cx_norm, cy_norm = lm.x, lm.y #lm.x*width, lm.y*height
            cx, cy = lm.x*self.W, lm.y*self.H
            coordX.append(cx)
            coordY.append(cy) 
            # coordX_norm.append(cx_norm)
            # coordY_norm.append(cy_norm) 
            
        self.X.append(coordX)
        self.Y.append(coordY)
        
        # self.X_norm.append(coordX_norm)
        # self.Y_norm.append(coordY_norm)
        
        endX = time.time()
        try:
            velX = self.Vel(np.array(self.X[-1])-np.array(self.X[-2]), endX-self.startX)
            velX_norm = self.Vel(np.array(self.X_norm[-1])-np.array(self.X_norm[-2]), endX-self.startX)
        except:
            velX = self.Vel(np.zeros(21), endX-self.startX)
            velX_norm = self.Vel(np.zeros(21), endX-self.startX)
            
        self.startX = time.time()

        endY = time.time()    
        try:
            velY = self.Vel(np.array(self.Y[-1])-np.array(self.Y[-2]), endY-self.startY)
        except:
            velY = self.Vel(np.zeros(21), endY-self.startY)
        self.startY = time.time()
            
            
        testX = np.array([velX[8], velX[12], velX[16]])
        testY = np.array([velY[8], velY[12], velY[16]])
        
        th = 0
        ang = np.arctan((coordY[12]-coordY[0])/(coordX[12]-coordX[0]))*180/np.pi
        if ang < 0:
            ang = 180 + ang
            
        self.ang = ang
                
#         len_ = np.sqrt((coordY[12]-coordY[0]) **2 + (coordX[12]-coordX[0])**2)
#         self.Len.append(len_)
#         try:
#             deltalen = abs(self.Len[-2] - self.Len[-1])
#         except: deltalen = 10

#         lenth = 4
#         if abs(velX_norm[0]) < 0.1 and abs(velX_norm[12]) > 0.1:
#             if np.all(testX > -th)  and np.all(testY > -th) and velX[0] < th: # and velY[0] < th:
#                 if deltalen < lenth:
#                     self.ang = ang

#             elif np.all(testX < th)  and np.all(testY < th) and velX[0] > -th: # and velY[0] > -th:
#                 if deltalen < lenth:
#                     self.ang = ang

#             elif np.all(testX > -th)  and np.all(testY < th) and velX[0] < th: # and velY[0] > -th:
#                 if deltalen < lenth:
#                     self.ang = ang

#             elif np.all(testX < th)  and np.all(testY > -th) and velX[0] > -th: # and velY[0] < th:
#                 if deltalen < lenth:
#                     self.ang = ang
            
            
            
    @staticmethod
    def Get_Roll_Pitch(self):
        AngleOY = None; AngleOZ = None
        cord = []
        for id, lm in enumerate(self.handLms.landmark):
            cx, cy = lm.x*self.W, lm.y*self.H

            d = None
            if cy >= 0 and cy < self.H and cx >= 0 and cx < self.W :
                d = self.depth_image[int(cy), int(cx)]

            cord.append((id, cx, cy, d))

        depth_mask = self.green_landmarks(self, cord)

        # new_mask = depth_mask.copy()
        OneD = depth_mask.flatten()
        NoZeros = OneD[OneD != 0]
        # fltred = NoZeros.copy()
        # NoZeros_sort = NoZeros.copy()
        NoZeros.sort()

        test = np.diff(NoZeros)

        try:
            ind = test[test>50][0]
            check = True
        except:
            check = False
        if check:
            where = np.where(test == ind)
            th = NoZeros[where[0][0]+1]
            # fltred = NoZeros[NoZeros < th]
            depth_mask[depth_mask >= th] = 0

        X = np.where(depth_mask!=0)[0]
        Y = np.where(depth_mask!=0)[1]
        Z = np.array([depth_mask[X[i], Y[i]] for i in range(len(X))])

        tmp_A = []
        tmp_b = []
        for i in range(len(X)):
            tmp_A.append([X[i], Y[i], 1])
            tmp_b.append(Z[i])
        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)

        fit, residual, rnk, s = lstsq(A, b)
        OYa1, OYb1, OYc1 = 0, 1, 0
        OZa1, OZb1, OZc1 = 1, 0, 0

        a2 = fit[0][0]
        b2 = fit[1][0]
        c2 = 1
        AngleOY = self.distance(self,OYa1,OYb1,OYc1,a2,b2,c2)
        AngleOZ = self.distance(self,OZa1,OZb1,OZc1,a2,b2,c2)

        return AngleOY, AngleOZ
    
    @staticmethod
    def FindTheAngle(self, cord):
        point1 = cord[-2]
        point2 = cord[-1]
        x_values = [point1[1], point2[1]]
        y_values = [point1[2], point2[2]]

        deltaY = y_values[0] - y_values[1]
        deltaX = x_values[0] - x_values[1]

        angleInDegrees = np.arctan(deltaY / deltaX) * 180 / np.pi
        if angleInDegrees < 0: 
            angleInDegrees = 179 + angleInDegrees
        return angleInDegrees
    
    
    def OrientStart(self, color_image, depth_image, handLms):
        self.color_image = color_image
        self.depth_image = depth_image
        self.handLms = handLms


        roll, pitch = self.Get_Roll_Pitch(self)
        self.Get_Yaw(self)
        
        print("R:", int(roll), "P:", int(pitch), "Y:", int(self.ang))
