# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 20:39:14 2021

@author: Philippe Savard
"""

""" IMPORTS """
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import Cascade
from imutils.video import WebcamVideoStream
#from imutils import face_utils
#import dlib

cv2.namedWindow("Facial Emotion Detection")

""" LBP FACE DETECTOR"""

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

"""WEBCAM INPUT"""

webcam = WebcamVideoStream(src=0).start()

# Reduces lag for face detection. Higher values reduces lag significantly but also decrease accuracy
RES_FACTOR = 4 

while True:
    #Read frames from the webcam 
    frame = webcam.read()
    
    # Extracting original frame's dimensions
    h, w, c = frame.shape
    
    nCol = 0
    nRow = 0
    nWidth = w
    nHeight = h   
    
    # Resize the frame for better performances
    resizedFrame = cv2.resize(frame, (int(w/RES_FACTOR), int(h/RES_FACTOR)))
    grayFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
    
    key = cv2.waitKey(20) # wait for space key to start
    if True:#key == 32: # For 
        
        # Converting to array for computation
        img = np.array(grayFrame)

        detected = detector.detect_multi_scale(img,
                                           scale_factor=1.2,
                                           step_ratio=1,
                                           min_neighbour_number=10,
                                           min_size=(30, 30),
                                           max_size=(220, 220))
        
        # Draw a rectangle around the face region
        for d in detected:
            nCol = d["c"] * RES_FACTOR
            nRow = d["r"] * RES_FACTOR
            nWidth = d["width"] * RES_FACTOR
            nHeight = d["height"] * RES_FACTOR
            
            cv2.rectangle(frame, (nCol, nRow), (nCol + nWidth, nRow + nHeight),(0, 255, 0), 2)
                  
    # Extract region of interest (the face) from the img             
    roi = frame[nRow : nRow + nHeight, nCol : nCol + nWidth ]        
    
    cv2.imshow("Facial Emotion Detection", frame)
    cv2.imshow("FACE ROI", roi)
    
   
    #Wait for ESC key input to close application
    key = cv2.waitKey(20) 
    if key == 27: # exit on ESC
        break
    
    
webcam.stop()

cv2.destroyWindow("Facial Emotion Detection")
cv2.destroyWindow("FACE ROI")