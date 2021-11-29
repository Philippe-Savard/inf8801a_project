# -*- coding: utf-8 -*-
import os
import glob
import cv2
import dlib
import numpy as np
from datetime import datetime
from skimage.feature import hog


# DLIB KEYPOINTS PREDICTOR 
p = "../shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)



TRAIN_IMAGE_PATH = 'ExtractedData/train/images.npy'
TRAIN_LABELS_PATH = 'ExtractedData/train/labels.npy'
TRAIN_LANDMARKS_PATH = 'ExtractedData/train/landmarks.npy'
TRAIN_HOG_PATH = 'ExtractedData/train/hog.npy'

TEST_IMAGE_PATH = 'ExtractedData/test/images.npy'
TEST_LABELS_PATH = 'ExtractedData/test/labels.npy'
TEST_LANDMARKS_PATH = 'ExtractedData/test/landmarks.npy'
TEST_HOG_PATH = 'ExtractedData/test/hog.npy'

# Emotions dictionnary
emotions = dict({"angry":0,
                 "disgust":1, 
                 "fear":2, 
                 "happy":3, 
                 "neutral":4,
                 "sad":5, 
                 "surprise":6})


def get_landmarks(image, rects):
    # Function copied from the stackoverflow post :
    # https://stackoverflow.com/questions/37689281/how-to-overlay-a-numpy-matrix-of-points-onto-an-image-in-opencv 
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def compute_HOG(image):
    fd = hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, feature_vector = True, channel_axis=-1)
    return fd



print("[INFO] Creating missing directories...")
if not os.path.exists('ExtractedData'):
    os.mkdir('ExtractedData')
    
if not os.path.exists('ExtractedData/train'):    
    os.mkdir('ExtractedData/train')
    
if not os.path.exists('ExtractedData/test'):
    os.mkdir('ExtractedData/test')
    
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


train_paths = glob.glob('FER2013/train'+"/*/*")

images = []
landmarks = []
labels = []
hogs = []

start_time = datetime.now()

print("[INFO] Extracting data from training images...")
print("[INFO] Extraction started at :", start_time )
for path in train_paths:
    
    img = cv2.imread(path)
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    landmark = get_landmarks(grayFrame, face_rects) / 48
    
    emotion_name = os.path.basename(os.path.dirname(path))
    outputLabel = np.zeros((7))
    outputLabel[emotions[emotion_name]] = 1
    
    hog_features = compute_HOG(img)
    
    images.append(grayFrame)
    landmarks.append(landmark) 
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

test_paths = glob.glob('FER2013/test'+"/*/*")

images = []
landmarks = []
labels = []
hogs = []

start_time = datetime.now()

print("[INFO] Extracting data from testing images...")
print("[INFO] Extraction started at :", start_time )
for path in test_paths:
    
    img = cv2.imread(path)
    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    landmark = get_landmarks(grayFrame, face_rects) / 48
    
    emotion_name = os.path.basename(os.path.dirname(path))
    outputLabel = np.zeros((7))
    outputLabel[emotions[emotion_name]] = 1

    hog_features = compute_HOG(img)

    images.append(grayFrame)
    landmarks.append(landmark) 
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