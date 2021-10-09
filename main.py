# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 20:39:14 2021

@author: Philippe Savard
"""

""" IMPORTS """
import cv2
import numpy as np

from skimage import data
from skimage.feature import Cascade


cv2.namedWindow("Facial Emotion Detection")

""" LBP FACE DETECTOR"""

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

"""WEBCAM INPUT"""

webcam = cv2.VideoCapture(0)

if webcam.isOpened(): 
    # try to get the first frame
    success, frame = webcam.read()
else:
    success = False

"""FRAME PROCESSING"""
while success:
    #Read frames from the webcam 
    success, frame = webcam.read()
    
    #Wait for ESC key input to close application
    key = cv2.waitKey(20) 
    if key == 27: # exit on ESC
        break
    
    img = np.array(frame)

    detected = detector.detect_multi_scale(img,
                                           scale_factor=1.1,
                                           step_ratio=1,
                                           min_neighbour_number=20,
                                           min_size=(30, 30),
                                           max_size=(220, 220))
    
    
    for d in detected:
        cv2.rectangle(frame, (d["c"], d["r"]), (d["c"] + d["width"], d["r"] + d["height"]), (0, 255, 0), 2)
        
    cv2.imshow("Facial Emotion Detection", frame)

webcam.release()

cv2.destroyWindow("Facial Emotion Detection")