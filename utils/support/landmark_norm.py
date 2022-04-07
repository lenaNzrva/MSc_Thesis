import copy
import cv2
import numpy as np
import itertools

class landmarkNorm:
    def __init__(self, landmarks, w=640, h=480):
        self.landmarks = landmarks
        self.w = w
        self.h = h
        
    @staticmethod
    def calc_landmark_list(self):
        landmark_point = []

        for _, landmark in enumerate(self.landmarks.landmark):
            landmark_x = min(int(landmark.x * self.w), self.w - 1)
            landmark_y = min(int(landmark.y * self.h), self.h - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    
    
    def pre_process_landmark(self):
        landmark_list = self.calc_landmark_list(self)
        
        ## Нормализация координат
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        NormData = list(map(normalize_, temp_landmark_list))


        ## Нормализация длин векторов
        temp_landmark_list = copy.deepcopy(landmark_list)
        New = []

        ## Большой палец
        for i in range(4):
            X = temp_landmark_list[i][0] - temp_landmark_list[i+1][0]
            Y = temp_landmark_list[i][1] - temp_landmark_list[i+1][1]
            New.append([X,Y])


        ## Указательный палец    
        X = temp_landmark_list[0][0] - temp_landmark_list[5][0]
        Y = temp_landmark_list[0][1] - temp_landmark_list[5][1]
        New.append([X,Y])

        for i in range(5,8):
            X = temp_landmark_list[i][0] - temp_landmark_list[i+1][0]
            Y = temp_landmark_list[i][1] - temp_landmark_list[i+1][1]
            New.append([X,Y])


        ## Средний палец
        X = temp_landmark_list[0][0] - temp_landmark_list[9][0]
        Y = temp_landmark_list[0][1] - temp_landmark_list[9][1]
        New.append([X,Y])

        for i in range(9,12):
            X = temp_landmark_list[i][0] - temp_landmark_list[i+1][0]
            Y = temp_landmark_list[i][1] - temp_landmark_list[i+1][1]
            New.append([X,Y])


        ## Безымянный палец
        X = temp_landmark_list[0][0] - temp_landmark_list[13][0]
        Y = temp_landmark_list[0][1] - temp_landmark_list[13][1]
        New.append([X,Y])

        for i in range(13,16):
            X = temp_landmark_list[i][0] - temp_landmark_list[i+1][0]
            Y = temp_landmark_list[i][1] - temp_landmark_list[i+1][1]
            New.append([X,Y])

        ## Мизинец
        X = temp_landmark_list[0][0] - temp_landmark_list[17][0]
        Y = temp_landmark_list[0][1] - temp_landmark_list[17][1]
        New.append([X,Y])

        for i in range(17,20):
            X = temp_landmark_list[i][0] - temp_landmark_list[i+1][0]
            Y = temp_landmark_list[i][1] - temp_landmark_list[i+1][1]
            New.append([X,Y])


        New_array = np.array(New)
        D = np.sqrt((New_array[:,0]**2) + (New_array[:,1]**2))
        D_norm = D/max(D)
        D_norm = D_norm.tolist()

        for d in D_norm:
            NormData.append(d)

        return NormData