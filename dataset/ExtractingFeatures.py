"""
Contains the necessary functions to extract in .npy file format all of the required data to train the
models.

@authors Philippe Savard & Gaspard Petitclair

"""
import os
import glob
import cv2
import dlib
import numpy as np
from datetime import datetime
from skimage.feature import hog
from os import path

# DLIB KEYPOINTS PREDICTOR 
p = "shape_predictor_68_face_landmarks.dat"
predictor = None
if path.exists(p):
    predictor = dlib.shape_predictor(p)
else:
    print("[WARNING] a shape_predictor_68_face_landmarks.dat not found. You can download this file at https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat")


# Training paths
TRAIN_IMAGE_PATH = 'dataset/ExtractedData/train/images.npy'
TRAIN_LABELS_PATH = 'dataset/ExtractedData/train/labels.npy'
TRAIN_LANDMARKS_PATH = 'dataset/ExtractedData/train/landmarks.npy'
TRAIN_HOG_PATH = 'dataset/ExtractedData/train/hog.npy'

# Testing paths
TEST_IMAGE_PATH = 'dataset/ExtractedData/test/images.npy'
TEST_LABELS_PATH = 'dataset/ExtractedData/test/labels.npy'
TEST_LANDMARKS_PATH = 'dataset/ExtractedData/test/landmarks.npy'
TEST_HOG_PATH = 'dataset/ExtractedData/test/hog.npy'

# Emotions dictionnary
emotions = dict({"angry":0,
                 "disgust":1, 
                 "fear":2, 
                 "happy":3, 
                 "neutral":4,
                 "sad":5, 
                 "surprise":6})


def get_landmarks(image, rects):
    """
        This function extract the face landmarks from the given image
        
        Parameters
        ----------
        image : ndarray
            The image to extract the features from.
        rects : dlib.rectangle
            The region of extraction given by a rectangle
        
        Returns
        -------
        Returns the matrix of points corresponding to the landmarks on the face.
    """
    # Function copied from the stackoverflow post :
    # https://stackoverflow.com/questions/37689281/how-to-overlay-a-numpy-matrix-of-points-onto-an-image-in-opencv 
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    if predictor == None:
        print("test1")
        raise BaseException("MissingShapePredictor")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def compute_HOG(image):
    """
        This function computes the HOG of an image.
        
        Parameters
        ----------
        image : ndarray
            The image to extract the features from.
        
        Returns
        -------
        Returns the descriptor containing the HOG
    """
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=False, feature_vector = True)
    return fd


def extract_data(write_over_data):
    """
        This function extract the data from the images in the current images database (FER2013). It extracts 
        image content, facial landmarks and histogram of oriented gradients.
        
        Parameters 
        ----------
        write_over_data : Boolean
            If this variable is set to True, any currently extracted data will be removed and re-extracted.
            
        Returns
        -------
        Returns the success value (True if the extraction succeeded and False if it failed).
    """
    try:
        # Checking if the files have already been extracted and we do not want to overwrite the data
        if not write_over_data and (os.path.exists(TRAIN_IMAGE_PATH) 
                               and os.path.exists(TEST_IMAGE_PATH)):
            print("[INFO] Data already extracted in a previous execution.")
            return True
        
        else:
            # Check if the dataset exists
            if not( os.path.exists('dataset/FER2013/train') and os.path.exists('dataset/FER2013/test')):
                print("[ERROR] No data to extract.")
                return False
            # Check if the predictor exist
            if predictor != None:
                
                if not os.path.exists('dataset/ExtractedData'):
                    print("[INFO] Creating missing directories...")
                    os.mkdir('dataset/ExtractedData')
                    
                if not os.path.exists('dataset/ExtractedData/train'):    
                    os.mkdir('dataset/ExtractedData/train')
                    
                if not os.path.exists('dataset/ExtractedData/test'):
                    os.mkdir('dataset/ExtractedData/test')
                    
                print("[INFO] Removing data from the current extraction files...")  
                
                # Erasing all of the file's data
                file = open(TRAIN_IMAGE_PATH, "w")
                file.close()
                file = open(TRAIN_LABELS_PATH, "w")
                file.close()
                file = open(TRAIN_LANDMARKS_PATH, "w")
                file.close()
                file = open(TRAIN_HOG_PATH, "w")
                file.close()
                file = open(TEST_IMAGE_PATH, "w")
                file.close()
                file = open(TEST_LABELS_PATH, "w")
                file.close()
                file = open(TEST_LANDMARKS_PATH, "w")
                file.close()
                file = open(TEST_HOG_PATH, "w")
                file.close()
                
                # List of all the images path's in the training directory
                train_paths = glob.glob('dataset/FER2013/train'+"/*/*")
                
                images = []
                landmarks = []
                labels = []
                hogs = []
                
                start_time = datetime.now()
                
                print("[INFO] Extracting data from training images...")
                print("[INFO] Extraction started at :", start_time )
                for file in train_paths:
                    
                    img = cv2.imread(file)
                    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                     
                    landmark = get_landmarks(grayFrame, face_rects) / 48
                    landmarks.append(landmark) 
                    
                    emotion_name = os.path.basename(os.path.dirname(file))
                    outputLabel = np.zeros((7))
                    outputLabel[emotions[emotion_name]] = 1
                    
                    hog_features = compute_HOG(grayFrame)
                    
                    images.append(grayFrame)
                    labels.append(outputLabel)
                    hogs.append(hog_features)
                
                images = [ x / 255 for x in images]
                images = [ x.reshape(48,48,1) for x in images]
                
                np.save(TRAIN_IMAGE_PATH, images)
                np.save(TRAIN_LANDMARKS_PATH, landmarks)
                np.save(TRAIN_HOG_PATH, hogs)
                np.save(TRAIN_LABELS_PATH, labels)
                
                end_time = datetime.now()
                
                print("[INFO] Done with training images.")
                print("[INFO] Extraction ended at :", end_time )
                print("[INFO] Total training data extraction time :", (end_time - start_time))
                
                test_paths = glob.glob('dataset/FER2013/test'+"/*/*")
                
                images = []
                landmarks = []
                labels = []
                hogs = []
                
                start_time = datetime.now()
                
                print("[INFO] Extracting data from testing images...")
                print("[INFO] Extraction started at :", start_time )
                for file in test_paths:
                    
                    img = cv2.imread(file)
                    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    
                    landmark = get_landmarks(grayFrame, face_rects) / 48
                    landmarks.append(landmark) 
                        
                    emotion_name = os.path.basename(os.path.dirname(file))
                    outputLabel = np.zeros((7))
                    outputLabel[emotions[emotion_name]] = 1
                
                    hog_features = compute_HOG(grayFrame)
                
                    images.append(grayFrame)
                    
                    labels.append(outputLabel)
                    hogs.append(hog_features)
                
                images = [ x / 255 for x in images]
                images = [ x.reshape(48,48,1) for x in images]
                
                np.save(TEST_IMAGE_PATH, images)
                np.save(TEST_LANDMARKS_PATH, landmarks)
                np.save(TEST_HOG_PATH, hogs)
                np.save(TEST_LABELS_PATH, labels)
                
                end_time = datetime.now()
                
                print("[INFO] Done with testing images.")
                print("[INFO] Extraction ended at :", end_time )
                print("[INFO] Total testing data extraction time :", (end_time - start_time))
                
                print("[INFO] Data extracted successfully :)")
                
                return True
            else:
                 print("[ERROR] shape_predictor_68_face_landmarks.dat not found. You can download this file at https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat")
                 return False
    except :
        print("[ERROR] An unexpected error occurred while extracting data :( ")
        return False