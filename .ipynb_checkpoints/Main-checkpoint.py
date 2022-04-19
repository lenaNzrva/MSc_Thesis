import cv2
# import copy
# import itertools
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
from utils.fps import CvFpsCalc
# from GestureRecognition.model import KeyPointClassifier

from utils.setup.setup import SetUp
from utils.support import*
from utils.position.pose_estimation import PoseEstimation
from utils.orientation.OrientEstimation import OrientEstimation

def main():

    hands, mpDraw, mpHands = SetUp.mediapipe()
    W = 640
    H = 480
    pipeline, config, align = SetUp.camera(640, 480, 60)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    poseEstimation = PoseEstimation(640, 480)
    orientEstimation = OrientEstimation(640, 480)
    
    
    try:
        pipeline.start(config)
    except:
        pipeline.stop()
        pipeline.start(config)
    
    while True:
        fps = cvFpsCalc.get()

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.flip(color_image, 1)
        dr_info = DrawInfo(color_image)

        coordinates = hands.process(color_image).multi_hand_landmarks
        if coordinates:
            list_for_coordinates = []
            for handLms in coordinates:
                rect = Bounding_Rect(color_image, handLms)
                brect, image = rect.draw()

            mpDraw.draw_landmarks(color_image, handLms, mpHands.HAND_CONNECTIONS)

            poseEstimation.PoseStart(handLms, depth_image)
            # orientEstimation.OrientStart(color_image, depth_image, handLms)


        dr_info.FPS(fps)
        cv2.imshow('RealSense', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            pipeline.stop()
            break

            
if __name__ == '__main__':
    main()