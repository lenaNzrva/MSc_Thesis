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
        
        self.ang = 90
        
        self.startX = time.time()
        self.startY = time.time()
        
    @staticmethod
    def interpolate_pixels_along_line(x0, y0, x1, y1):
        pixels = []
        steep = abs(y1 - y0) > abs(x1 - x0)

        # Ensure that the path to be interpolated is shallow and from left to right
        if steep:
            t = x0
            x0 = y0
            y0 = t

            t = x1
            x1 = y1
            y1 = t

        if x0 > x1:
            t = x0
            x0 = x1
            x1 = t

            t = y0
            y0 = y1
            y1 = t

        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx  # slope

        # Get the first given coordinate and add it to the return list
        x_end = round(x0)
        y_end = y0 + (gradient * (x_end - x0))
        xpxl0 = x_end
        ypxl0 = round(y_end)
        if steep:
            pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
        else:
            pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

        interpolated_y = y_end + gradient

        # Get the second given coordinate to give the main loop a range
        x_end = round(x1)
        y_end = y1 + (gradient * (x_end - x1))
        xpxl1 = x_end
        ypxl1 = round(y_end)

        # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
        for x in range(xpxl0 + 1, xpxl1):
            if steep:
                pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

            else:
                pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

            interpolated_y += gradient

        # Add the second given coordinate to the given list
        if steep:
            pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
        else:
            pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

        return pixels
    
    
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
        ang = np.arctan((self.cord[12][1]-self.cord[0][1])/(self.cord[12][0]-self.cord[0][0]))*180/np.pi
        if ang < 0:
            ang = 180 + ang
            
        self.ang = ang
        
        
    @staticmethod
    def Get_Roll_Pitch(self):
        AngleOY = None; AngleOZ = None

        CONNECTIONS = [(3, 4),
                         (0, 5),
                         (17, 18),
                         (0, 17),
                         (13, 14),
                         (13, 17),
                         (18, 19),
                         (5, 6),
                         (5, 9),
                         (14, 15),
                         (0, 1),
                         (9, 10),
                         (1, 2),
                         (9, 13),
                         (10, 11),
                         (19, 20),
                         (6, 7),
                         (15, 16),
                         (2, 3),
                         (11, 12),
                         (7, 8)]

        k = 1
        knn = np.linspace(-k,k,k*2+1, dtype="int")

        pixels_list = []
        for connections in CONNECTIONS:
            pixels = self.interpolate_pixels_along_line(self.cord[connections[0]][0], self.cord[connections[0]][1], self.cord[connections[1]][0], self.cord[connections[1]][1])

            for p in pixels:
                for i in knn:
                    pixels_list.append([p[0]+i, p[1]+i])

        pixels_list = np.array(pixels_list)
        
        Z = np.array([self.depth_image[p[1],p[0]] for p in pixels_list])
        X = pixels_list[:,1]
        Y = pixels_list[:,0]

        X = X[Z != 0]
        Y = Y[Z != 0]
        Z = Z[Z != 0]

        Z_copy = Z.copy()
        Z_copy.sort()
        
        test = np.diff(Z_copy)
        
        if test[np.argmax(test)] > 50:
            max_ind = np.argmax(test)
            th = Z_copy[max_ind+1]
            
            X = X[Z < th]
            Y = Y[Z < th]
            Z = Z[Z < th]


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
    
    
    def OrientStart(self, color_image, depth_image, handLms):
        self.color_image = color_image
        self.depth_image = depth_image
        self.handLms = handLms
        
        self.cord = []
        for id, lm in enumerate(self.handLms.landmark):
            cx, cy = lm.x*self.W, lm.y*self.H
            
            self.cord.append((cx, cy))

        roll, pitch = self.Get_Roll_Pitch(self)
        self.Get_Yaw(self)
        
        print("R:", int(roll), "P:", int(pitch), "Y:", int(self.ang))
