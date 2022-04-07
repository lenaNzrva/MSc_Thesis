import cv2
import numpy as np

class Bounding_Rect():
    
    def __init__(self, image, landmarks):
        self.image = image
        self.landmarks = landmarks
        
        image_width, image_height = self.image.shape[1], self.image.shape[0]
        landmark_array = np.empty((0, 2), int)
        
        for _, landmark in enumerate(self.landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)
        self.brect = [x, y, x + w, y + h]
        
    
    def draw(self):   
        
        cv2.rectangle(self.image, (self.brect[0], self.brect[1]), (self.brect[2], self.brect[3]),(0, 0, 0), 1)
        
        return self.brect , self.image