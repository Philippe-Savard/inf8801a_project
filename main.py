"""

@authors Philippe Savard & Gaspard Petitclair

"""

import cv2
import numpy as np
from imutils import face_utils
import dlib
from neuralnetwork.EmotionsNetwork import EmotionsNetwork
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from os import path

PLOT_MODELS = False

# Classified emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy",
            "Neutral", "Sad", "Surprise", "NONE"]

def print_stats(history, name):
    
    """
        This function plots the resulting history of the trained model. It shows the function of accuracy per epoch
        and loss per epoch.
        
        Parameters
        ----------
        history
            The model history to be displayed
        name : str
            The name of the model to be displayed
    """
    if not history == None:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(name + 'model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(name + 'model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        print("[INFO] No data to plot. Please retrain the " + name +" model to obtain the history.")


print("[INFO] Scanning for trained neural networks.")

# Build network
cnn = EmotionsNetwork()


print("[INFO] Loading CNN Only model.")
cnn_success, cnn_only_model = cnn.get_cnn_only_model()
if not cnn_success:
    print("[WARNING] Missing training data for CNN Only model.")
else :
    cnn_only_model.summary()
    print_stats(cnn.history_cnn_only, 'cnn_only')

print("[INFO] Loading CNN + Landmarks model.")
landmarks_success, cnn_landmarks_model = cnn.get_cnn_landmarks_model()
if not landmarks_success:
    print("[WARNING] Missing training data for CNN + Landmarks model.")
else:
    cnn_landmarks_model.summary()
    print_stats(cnn.history_cnn_landmarks, 'cnn_with_landmarks')

print("[INFO] Loading CNN +Landmarks + HOG model.")
hog_success, cnn_landmarks_hog_model = cnn.get_cnn_landmarks_hog_model()
if not hog_success:
    print("[WARNING] Missing training data for CNN + Landmarks + HOG model.")
else :
    cnn_landmarks_hog_model.summary()
    print_stats(cnn.history_cnn_landmarks_hog, 'cnn_with_landmarks_hog')

# Saves the plotted image of all the network's models
if PLOT_MODELS and cnn_success and landmarks_success and hog_success:
    plot_model(cnn_landmarks_model, to_file='cnn_with_landmarks.png', show_shapes=True, show_layer_names=True)
    plot_model(cnn_only_model, to_file='cnn_only_model.png', show_shapes=True, show_layer_names=True)
    plot_model(cnn_landmarks_hog_model, to_file='cnn_with_landmarks_hog.png', show_shapes=True, show_layer_names=True)

# DLIB FACE DETECTOR AND KEYPOINTS PREDICTOR
detector = dlib.get_frontal_face_detector()

p = "shape_predictor_68_face_landmarks.dat"
predictor = None
if path.exists(p):
    predictor = dlib.shape_predictor(p)
else:
    print("[WARNING] shape_predictor_68_face_landmarks.dat not found. You can download this file at https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat")

# WEBCAM INPUT
webcam = cv2.VideoCapture(0)

print("[INFO] Starting Emotion Detection. Press ESC to stop the application...")

# Create the main frame window
cv2.namedWindow("Facial Emotion Detection")

while True:
    try:
        # Read frames from the webcam
        _, frame = webcam.read()
        
        # Making sure the shape predictor exist (for resizing)
        if predictor != None:
            
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
                    if minX > x:
                        minX = x
                    if maxX < x:
                        maxX = x
                    if minY > y:
                        minY = y
                    if maxY < y:
                        maxY = y
                        
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    
                # Draw the boundary around the face
                cv2.rectangle(frame, (minX, maxY), (maxX, minY), (0, 255, 0), 1)
        
                # Extract region of interest (the face) from the frame
                roi = grayFrame[minY: maxY, minX: maxX]
        
                # Check if a face has been extracted from the frame
                if len(roi) > 0:
        
                    # Resizing the input for it to be 48x48
                    resizedRoi = cv2.resize(roi, (48, 48))
        
                    # Normalizing the region using MinMax normalization
                    normRoi = cv2.normalize(resizedRoi, resizedRoi, np.min(resizedRoi), np.max(resizedRoi), cv2.NORM_MINMAX)
                    unitRoi = normRoi / 255.0
        
                    # Reshape the ROI to fit the network input shape
                    reshapedRoi = unitRoi.reshape(1, 48, 48, 1)
                    
                    emotionsProb = [0, 0, 0, 0, 0, 0, 0, 1]
                    
                    if cnn_success:
                        # Get the probability of each classified emotions from the network
                        emotionsProb = cnn_only_model.predict(reshapedRoi)
                        
                    # Extract the index of the highest probability
                    result = np.argmax(emotionsProb)
        
                    # Display the emotion as text on the screen
                    cv2.putText(frame, EMOTIONS[result], (minX, maxY + 18), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        
                    # Display the face region after normalization
                    cv2.imshow("FACE ROI #" + str(i), normRoi)
            
        else:
            # Warning if missing the predictor file
            cv2.putText(frame, "Oops! Please download the shape_predictor_68_face_landmarks.dat file", (0, 100), cv2.FONT_HERSHEY_PLAIN, 1.00, (0, 0, 255), 2)
       
        # Display the emotion as text on the screen
        cv2.putText(frame, "Press [ESC] to quit", (0, 18), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Facial Emotion Detection", frame)

        # Wait for ESC key input to close application
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            print("[INFO] Received stop instruction. Stopping execution now...")
            break
    except:
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            print("[INFO] Received stop instruction. Stopping execution now...")
            break
        
        print("[ERROR] An unexpected error occurred. Press [ESC] to quit")
        continue

webcam.release()
cv2.destroyAllWindows()

print("[INFO] End of execution. Have a nice day :)")
