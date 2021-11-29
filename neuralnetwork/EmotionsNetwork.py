"""
This file contains the CNN network class EmotionsNetwork used to build and train the models used in the paper.

@author Philippe Savard

"""

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import load_model
from datetime import datetime
from os import path
from keras.models import Model
import numpy as np

NB_EPOCHS = 13
BATCH_SIZE = 128
TARGET_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)
CLASS_MODE = 'categorical'
PIXEL_SCALE = 1 / 255
LOSS = 'categorical_crossentropy'
COLOR_MODE = 'grayscale'

CNN_ONLY_PATH = 'neuralnetwork/models/cnn_only_model'
CNN_LANDMARKS_PATH = 'neuralnetwork/models/cnn_landmarks_model'
CNN_LANDMARKS_HOG_PATH = 'neuralnetwork/models/cnn_landmarks_hog_model'

TRAIN_IMAGES_PATH = 'dataset/ExtractedData/train/images.npy'
TRAIN_LABELS_PATH = 'dataset/ExtractedData/train/labels.npy'

TRAIN_LANDMARKS_DATA_PATH = 'dataset/ExtractedData/train/landmarks.npy'
TRAIN_HOG_DATA_PATH = 'dataset/ExtractedData/train/hog.npy'

TEST_IMAGES_PATH = 'dataset/ExtractedData/test/images.npy'
TEST_LABELS_PATH = 'dataset/ExtractedData/test/labels.npy'

TEST_LANDMARKS_DATA_PATH = 'dataset/ExtractedData/test/landmarks.npy'
TEST_HOG_DATA_PATH = 'dataset/ExtractedData/train/hog.npy'


class EmotionsNetwork:
    
    """
        EmotionsNetwork objects contain the models for all three CNN used in the article. 
        
            (cnn_only_model) : is the model used as a baseline and only uses the images for training.
        
            (cnn_landmarks_model) : correspond to the baseline model with an
            additionnal landmarks inputs to the dense layers. 
        
            (cnn_landmarks_hog_model) : correspond to the cnn_landmarks_model
            with another additionnal hog descriptors inputs to the dense layers.
    """
    
    def __init__(self, trainingdata_path="dataset/FER2013/train/", testingdata_path="dataset/FER2013/test/"):
        
        """
            Default constructor for the EmotionsNetwork class.
        """
        
        self.history_cnn_only = None
        self.__train_images = np.load(TRAIN_IMAGES_PATH)
        self.__train_labels = np.load(TRAIN_LABELS_PATH)
        
        self.__test_images = np.load(TEST_IMAGES_PATH)
        self.__test_labels = np.load(TEST_LABELS_PATH)
        
        self.__train_landmarks = np.load(TRAIN_LANDMARKS_DATA_PATH)
        self.__test_landmarks = np.load(TEST_LANDMARKS_DATA_PATH)
        
        self.__train_hog = np.load(TRAIN_HOG_DATA_PATH,allow_pickle=True)
        self.__test_hog = np.load(TEST_HOG_DATA_PATH,allow_pickle=True)

        # Build all the models """
        self.cnn_only_model = self.__build_cnn_only_model()
        self.cnn_landmarks_model = self.__build_cnn_landmarks_model()
        self.cnn_landmarks_hog_model =self.__build_cnn_landmarks_hog_model()
    
    #========================================================================================#
    #                                    CNN                                                 # 
    #========================================================================================#
    
    def __build_cnn_only_model(self):
        
        """
            This method build the sequential network model's for the CNN only baseline 
            approach. The model takes in a normalized grayscaled image resized to 48x48
            pixels dimensions. the output is a 7 emotions classes probability vector 
            obtained after classification.
        """  
        
        # CNN only Sequential model as illustrated in the paper.
        cnn_only_model = Sequential([
            # Input resized (48x48), normalized (MinMax norm.) and cropped(dlib) gray-scaled image 
            Input(shape=INPUT_SHAPE, name='ConvInput'),
            #
            Conv2D(64,(5,5), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
            Dropout(0.4),
            #
            Conv2D(128,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
            Dropout(0.4),
            #
            Conv2D(512,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
            Dropout(0.4),
            #
            Conv2D(512,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
            Dropout(0.4),
            Flatten(),
            # (out) Image matrix after 4 convolutionnal layers
            Dense(256, activation='relu'),
            #
            Dense(512, activation='relu'),
            #
            Dense(7, activation='softmax')])
        
        return cnn_only_model
    
    
    def train_cnn_only_model(self, save_path=CNN_ONLY_PATH):
            
        start_time = datetime.now()
        print("[INFO] Training cnn_only model. Starting time: ", start_time)
        
        # Compiling the model
        self.cnn_only_model.compile(loss=LOSS, optimizer="adam", metrics=["accuracy"])
        
        print("[INFO] Model compiled sucessfully!")
        
        # Trainning the model with the data
        self.cnn_only_model.fit(self.__train_images, self.__train_labels, 
                                validation_data=(self.__test_images, self.__test_labels),
                                epochs=NB_EPOCHS, 
                                batch_size=BATCH_SIZE)
        
        finish_time = datetime.now()
            
        print("[INFO] Training complete. End time: ", finish_time)
        print("[INFO] Total training time ", (finish_time - start_time))
        
        self.cnn_only_model.save(save_path)
        
        print("[INFO] Trained model saved at ", save_path)
        
    
    def get_cnn_only_model(self):
        
        """
            This method returns the trained cnn_only model. If the model has already been
            trained, it uses the trained model. Otherwise, it trains the current cnn_only model
            and returns it.
        """
        
        if path.exists(CNN_ONLY_PATH):
            return load_model(CNN_ONLY_PATH, compile=True)
        else:
            print("[INFO] No cnn_only model have been trained.")
            self.train_cnn_only_model()
            return self.cnn_only_model
    
    
    #========================================================================================#
    #                                    CNN + LANDMARKS                                     # 
    #========================================================================================#
    
    def __get_conv_layers(self, input_shape = INPUT_SHAPE):
        
        """
            This method builds the convolution branch of the model.
        """
        
        model_in = Input(shape=input_shape, name='ConvInput')
        
        conv1 = Conv2D(64,(5,5), activation='relu')(model_in)
        pooling1 = MaxPooling2D(pool_size=(2,2), strides=2)(conv1)
        
        conv2 = Conv2D(128,(3,3), activation='relu')(pooling1)
        pooling2 = MaxPooling2D(pool_size=(2,2), strides=2)(conv2)
        drop1 = Dropout(0.4)(pooling2)
        
        conv3 = Conv2D(512,(3,3), activation='relu')(drop1)
        pooling3 = MaxPooling2D(pool_size=(2,2), strides=2)(conv3)
        drop2 = Dropout(0.4)(pooling3)
        
        conv4 = Conv2D(512,(3,3), activation='relu')(drop2)
        pooling4 = MaxPooling2D(pool_size=(2,2), strides=2)(conv4)
        drop3 = Dropout(0.4)(pooling4)
        
        flat1 = Flatten()(drop3)
        model_out = Dense(128, activation='relu')(flat1)
        
        model_out = Dropout(0.4)(model_out)
        
        return model_in, model_out
    
    
    def __get_landmarks_layers(self, input_shape=(68,2)):
        
        """
            This method builds the landmarks branch of the model.
        """
        
        model_in = Input(shape=input_shape, name='LandmarksInput')
        
        dense1 = Dense(68, activation='relu')(model_in)
        flat1 = Flatten()(dense1)
        
        model_out = Dense(128, activation='relu') (flat1)
        model_out = Dropout(0.4)(model_out)
        return model_in, model_out
    
    
    def __build_cnn_landmarks_model(self):
        
        """
            This method build a non-linear network model for the CNN + landmarks approach. 
            The model takes has inputs a normalized grayscaled image resized to 48x48
            pixels dimensions and the 68 facials landmarks. the output is a 7 emotions 
            classes probability vector obtained after classification.
        """ 
        
        # Getting the inputs and outputs layers of the two branches
        conv_model_in, conv_model_out = self.__get_conv_layers()
        landmarks_model_in, landmarks_model_out = self.__get_landmarks_layers()
        
        # Merging the branches together
        merged_model1 = Concatenate()([landmarks_model_out, conv_model_out])
        
        # Flattening the merged model
        flat_merged = Flatten()(merged_model1)
        
        # 256 & 512 fully connected layers 
        dense1 = Dense(256, activation='relu')(flat_merged)
        dense2 = Dense(512, activation='relu')(dense1)
        
        # Model output classified into 7 emotions
        model_out = Dense(7, activation='softmax')(dense2)
        
        # Assemble the model
        cnn_landmarks_model = Model(inputs=[conv_model_in, landmarks_model_in], outputs=[model_out])
        
        return cnn_landmarks_model
    
    
    def train_cnn_landmarks_model(self, save_path=CNN_LANDMARKS_PATH):
        
        start_time = datetime.now()
        print("[INFO] Training cnn_landmarks model. Starting time: ", start_time)
        
        # Compiling the model
        self.cnn_landmarks_model.compile(loss=LOSS, optimizer="adam", metrics=["accuracy"])
        
        print("[INFO] Model compiled sucessfully!")
       
        # Trainning the model with the datasets
        self.cnn_landmarks_model.fit([self.__train_images, self.__train_landmarks], self.__train_labels, 
                                     validation_data=([self.__test_images, self.__test_landmarks], self.__test_labels),
                                     epochs=NB_EPOCHS, 
                                     batch_size=BATCH_SIZE)
        
        finish_time = datetime.now()
            
        print("[INFO] Training complete. End time: ", finish_time)
        print("[INFO] Total training time ", (finish_time - start_time))
        
        self.cnn_landmarks_model.save(save_path)
        
        print("[INFO] Trained model saved at ", save_path)
        
        
    def get_cnn_landmarks_model(self):
        
        """
            This method returns the trained cnn_landmarks model. If the model has already been
            trained, it uses the trained model. Otherwise, it trains the current cnn_landmarks model
            and returns it.
        """
        
        if path.exists(CNN_LANDMARKS_PATH):
            return load_model(CNN_LANDMARKS_PATH, compile=True)
        else:
            print("[INFO] No cnn_landmarks model have been trained.")
            self.train_cnn_landmarks_model()
        return self.cnn_landmarks_model
    
    
    
    #========================================================================================#
    #                                    CNN + LANDMARKS + HOG                               # 
    #========================================================================================#
    
    def __get_hog_layers(self, input_shape=(72,)):
        
        """
            This method builds the hog's branch of the model.
        """
        
        model_in = Input(shape=input_shape, name='HogInput')
        
        dense1 = Dense(140, activation='relu')(model_in)
        flat1 = Flatten()(dense1)
        
        model_out = Dense(128, activation='relu') (flat1)
        
        model_out = Dropout(0.4)(model_out)
        
        return model_in, model_out
    
    
    def __build_cnn_landmarks_hog_model(self):
        
        """
            This method build a non-linear network model for the CNN + landmarks + hog approach. 
            The model takes has inputs a normalized grayscaled image resized to 48x48
            pixels dimensions and the 68 facials landmarks. the output is a 7 emotions 
            classes probability vector obtained after classification.
        """ 
        
        # Getting the inputs and outputs layers of the two branches
        conv_model_in, conv_model_out = self.__get_conv_layers()
        landmarks_model_in, landmarks_model_out = self.__get_landmarks_layers()
        hog_model_in, hog_model_out = self.__get_hog_layers()
        
        # Merging the branches together
        merged_model1 = Concatenate()([hog_model_out,landmarks_model_out, conv_model_out])
        
        # Flattening the merged model
        flat_merged = Flatten()(merged_model1)
        
        # 256 & 512 fully connected layers 
        dense1 = Dense(256, activation='relu')(flat_merged)
        dense2 = Dense(512, activation='relu')(dense1)
        
        # Model output classified into 7 emotions
        model_out = Dense(7, activation='softmax')(dense2)
        
        # Assemble the model
        cnn_landmarks_hog_model = Model(inputs=[conv_model_in, landmarks_model_in, hog_model_in], outputs=[model_out])
        
        return cnn_landmarks_hog_model
    
    def train_cnn_landmarks_hog_model(self, save_path=CNN_LANDMARKS_HOG_PATH):
        
        start_time = datetime.now()
        print("[INFO] Training cnn_landmarks model. Starting time: ", start_time)
        
        # Compiling the model
        self.cnn_landmarks_hog_model.compile(loss=LOSS, optimizer="adam", metrics=["accuracy"])
        
        print("[INFO] Model compiled sucessfully!")
       
        # Trainning the model with the datasets
        self.cnn_landmarks_hog_model.fit([self.__train_images, self.__train_landmarks,self.__train_hog], self.__train_labels, 
                                     validation_data=([self.__test_images, self.__test_landmarks,self.__test_hog], self.__test_labels),
                                     epochs=NB_EPOCHS, 
                                     batch_size=BATCH_SIZE)
        
        finish_time = datetime.now()
            
        print("[INFO] Training complete. End time: ", finish_time)
        print("[INFO] Total training time ", (finish_time - start_time))
        
        self.cnn_landmarks_hog_model.save(save_path)
        
        print("[INFO] Trained model saved at ", save_path)

    def get_cnn_landmarks_hog_model(self):
        
        """
            This method returns the trained cnn_landmarks model. If the model has already been
            trained, it uses the trained model. Otherwise, it trains the current cnn_landmarks model
            and returns it.
        """
        
        if path.exists(CNN_LANDMARKS_HOG_PATH):
            return load_model(CNN_LANDMARKS_HOG_PATH, compile=True)
        else:
            print("[INFO] No cnn_landmarks model have been trained.")
            self.train_cnn_landmarks_hog_model()
        return self.cnn_landmarks_hog_model
