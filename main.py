# -*- coding: utf-8 -*-
"""
@author: Philippe Savard
"""
import cv2
import numpy as np
from imutils import face_utils
import dlib
from neuralnetwork.EmotionsNetwork import EmotionsNetwork
from tensorflow.keras.utils import plot_model

# Classified emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "NONE"]

print("[INFO] Scanning for trained neural networks...")

# Load trained model 
cnn = EmotionsNetwork()

cnn_only_model = cnn.get_cnn_only_model()
cnn_only_model.summary()

cnn_landmarks_model = cnn.get_cnn_landmarks_model()
cnn_landmarks_model.summary()



# DLIB FACE DETECTOR AND KEYPOINTS PREDICTOR 
detector = dlib.get_frontal_face_detector()

p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)

# WEBCAM INPUT
webcam = cv2.VideoCapture(0)

print("[INFO] Starting Emotion Detection. Press ESC to stop the application...")

# Create the main frame window
cv2.namedWindow("Facial Emotion Detection")

while True:
    
    # Read frames from the webcam 
    _, frame = webcam.read()
    
    # Converting the frame into the grayscale color space
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(grayFrame, 0)
    
    # For all faces in the frame
    for (i, face) in enumerate(faces):
        
        # Finding the facial landmarks 
        landmarks = predictor(grayFrame, face)
        
        # Converts the landmarks into a 2D numpy array of x, y coordinnates
        landmarks = face_utils.shape_to_np(landmarks)
        
        # Extract default landmarks
        minX = landmarks[i][0]
        maxX = landmarks[i][0]
        minY = landmarks[i][1]
        maxY = landmarks[i][1]
        
        # Draw the landmarks (dots) on the preview image
        for (x, y) in landmarks:
            
            # Update the keypoints boundaries
            if minX > x: minX = x
            if maxX < x: maxX = x
            if minY > y: minY = y
            if maxY < y: maxY = y
            
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # Draw the boundary around the face
        cv2.rectangle(frame,(minX, maxY),(maxX, minY),(0, 255, 0), 1)
        
        # Extract region of interest (the face) from the frame             
        roi = grayFrame[minY : maxY, minX : maxX]   
        
        # Check if a face has been extracted from the frame
        if len(roi) > 0:
            
            # Resizing the input for it to be 48x48 
            resizedRoi = cv2.resize(roi, (48,48))
    
            # Normalizing the region using MinMax normalization
            normRoi = cv2.normalize(resizedRoi, resizedRoi, np.min(resizedRoi), np.max(resizedRoi), cv2.NORM_MINMAX)
            unitRoi = normRoi / 255.0
            
            # Reshape the ROI to fit the network input shape            
            reshapedRoi = unitRoi.reshape(1,48,48,1)
            
            # Get the probability of each classified emotions from the network 
            emotionsProb = cnn_only_model.predict(reshapedRoi)
            
            # Extract the index of the highest probability
            result = np.argmax(emotionsProb)
            
            # Display the emotion as text on the screen 
            cv2.putText(frame, EMOTIONS[result], (minX, maxY + 18), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
            # Display the face region after normalization
            cv2.imshow("FACE ROI #" + str(i), normRoi)
    
    # Display the emotion as text on the screen 
    cv2.putText(frame,"Press [ESC] to quit", (0, 18), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Facial Emotion Detection", frame)
   
    # Wait for ESC key input to close application
    key = cv2.waitKey(20) 
    if key == 27: # exit on ESC
        print("[INFO] Received stop instruction. Stopping execution now...")
        break

webcam.release()
cv2.destroyAllWindows()

print("[INFO] End of execution. Have a nice day :)")