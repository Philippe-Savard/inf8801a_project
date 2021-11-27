# -*- coding: utf-8 -*-
import os
import glob
import cv2
import dlib
import numpy as np
from imutils import face_utils

# DLIB FACE DETECTOR AND KEYPOINTS PREDICTOR 
detector = dlib.get_frontal_face_detector()
p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

allpathstrain = glob.glob('dataset/FER2013/train'+"/*")
listofpaths = []
listofimageIndex = []
listofhistograms = []
    
for path in allpathstrain:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
    # Detect faces in the grayscale image
    faces = detector(img, 0)
        
     
    # For all faces in the frame
    for (i, face) in enumerate(faces):
         # Finding the facial landmarks 
         landmarks = predictor(img, face)
            
         # Converts the landmarks into a 2D numpy array of x, y coordinnates
         landmarks = face_utils.shape_to_np(landmarks)


allpathstrain = glob.glob('dataset/FER2013/test'+"/*")
listofpaths = []
listofimageIndex = []
listofhistograms = []

    
for path in allpathstrain:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
    # Detect faces in the grayscale image
    faces = detector(img, 0)
        
     
    # For all faces in the frame
    for (i, face) in enumerate(faces):
         # Finding the facial landmarks 
         landmarks = predictor(img, face)
            
         # Converts the landmarks into a 2D numpy array of x, y coordinnates
         landmarks = face_utils.shape_to_np(landmarks)


