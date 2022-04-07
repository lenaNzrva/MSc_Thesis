import cv2

class DrawInfo:
    def __init__(self, image):
        self.image = image

    def Text(self, brect, hand_sign_text):
        cv2.rectangle(self.image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        # info_text = handedness.classification[0].label[0:]
        # if hand_sign_text != "":
        info_text = hand_sign_text
        cv2.putText(self.image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return self.image
        
    def FPS(self, fps):
        cv2.putText(self.image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(self.image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return self.image
        