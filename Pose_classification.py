import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
# from pipline import *
from pipline2 import *
import time
import config as config
import pandas as pd

# Setup Pose function for video.

# mp_pose = mp.solutions.holistic

# pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
pose_video = mp_pose.Holistic(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.5, model_complexity=1)

# def video():
    # pose_video = mp_pose.Holistic(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)

    # Initialize the VideoCapture object to read from the webcam.

    # camera_video = cv2.VideoCapture(0)

# camera_video = cv2.VideoCapture(0)

    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/11.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/12.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/13.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/14.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/15.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/16.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/17.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/18.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/posto2.mp4')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/posto3.avi')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/posto4.avi')
    # camera_video = cv2.VideoCapture('c:/Users/lazaro.llins/Documents/App_New_Holistic/videos/posto5.avi')
    # camera_video.set(3,1280)
    # camera_video.set(4,960)



def finished():
    config.kill_cam = True
    # kill_cam = True
    return "kill_webcam"



def video():
    global Ang

    

    camera_video = cv2.VideoCapture(0)

    # Initialize a resizable window.
    # cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)


    # if not camera_video.isOpened():
    #     print("Camera video is not open")
    #     exit()

    # Iterate until the webcam is accessed successfully.
    
    while camera_video.isOpened():
        
        if config.kill_cam == True:
                        
                for i in range (10):
                    
                    print("kill no IF")
                
                break

    # while True:    
        # Read a frame.
        ok, frame = camera_video.read()
        
        # Check if frame is not read properly.
        if not ok:
            
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.

        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # frame = cv2.resize(frame, (int((frame_width*1/2) * (320 / (frame_height*1/2))), 320))
        

        t1 = time.time()
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(frame, pose_video, display=False)
        

        
        
        # Check if the landmarks are detected.
        if landmarks:
            
            # Perform the Pose Classification.
            frame, Ang = classifyPose(landmarks, frame, display=False)

           
           
            
            # print(Ang[0])
            
        # Calculating the FPS 

        t2 = time.time() - t1

        fps = 1/(t2)
    
        # Display FPS on the image

        cv2.putText(frame, "FPS: {:.0f} ".format(fps), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (203, 52, 247), 1)

        t1 = t2
        # Display the frame.
        cv2.imshow('Pose Classification Hands', frame)
        # print(Ang[0])
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed

      

        k = cv2.waitKey(10) & 0xFF
        # Check if 'ESC' is pressed.
        if(k == 27):
            # Break the loop.
            # return Ang[0]
            break
        
    # Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()
    return Ang[0]



