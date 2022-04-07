import pyrealsense2 as rs
import mediapipe as mp

class setup:
    def camera(self, width, height, max_fps):

        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        ##Set up resolution
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, max_fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, max_fps)
        align = rs.align(rs.stream.color)
        
        return pipeline, config, align
    
    def mediapipe(self, max_num_hands=1, min_det_conf=0.5, min_tr_conf=0.5):
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=max_num_hands, 
                              min_detection_confidence=min_det_conf, 
                              min_tracking_confidence=min_tr_conf) 
        mpDraw = mp.solutions.drawing_utils
        
        return hands, mpDraw, mpHands
    
SetUp = setup()