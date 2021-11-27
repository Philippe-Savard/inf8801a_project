"""
This file contains the CNN network class EmotionsNetwork used to build and train the models used in the paper.

@author Philippe Savard

"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import load_model
from datetime import datetime
from os import path


NB_EPOCHS = 1
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
        self.__train = ImageDataGenerator(rescale=PIXEL_SCALE)
        self.__test  = ImageDataGenerator(rescale=PIXEL_SCALE)
        
        # Training portion of the dataset contains approx. 80% of the complete dataset
        self._train_dataset = self.__train.flow_from_directory(trainingdata_path,
                                                               color_mode=COLOR_MODE,
                                                               target_size=TARGET_SIZE, 
                                                               batch_size=BATCH_SIZE, 
                                                               class_mode=CLASS_MODE)

        # Testing portion of the dataset contains approx. 20% of the complete dataset
        self._test_dataset = self.__test.flow_from_directory(testingdata_path, 
                                                             color_mode=COLOR_MODE,
                                                             target_size=TARGET_SIZE, 
                                                             batch_size=BATCH_SIZE, 
                                                             class_mode=CLASS_MODE)
        
        # Build all the models 
        self.cnn_only_model = self.__build_cnn_only_model()
        
    
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
            Conv2D(64,(5,5), activation='relu', input_shape=INPUT_SHAPE), 
            MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"),
            #
            Conv2D(128,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"),
            #
            Conv2D(512,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"),
            #
            Conv2D(512,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"),
            # (out) Image matrix after 4 convolutionnal layers
            Flatten(),
            #
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
        
        # Trainning the model with the datasets
        self.cnn_only_model.fit(self._train_dataset, 
                                epochs=NB_EPOCHS, 
                                batch_size=BATCH_SIZE, 
                                validation_data=self._test_dataset)
        
        finish_time = datetime.now()
            
        print("[INFO] Training complete. End time: ", finish_time)
        print("[INFO] Total training time ", (finish_time - start_time))
        
        self.cnn_only_model.save(save_path)
        
        print("[INFO] Trained model saved at ", save_path)
        
    
    def get_cnn_only_model(self):
        
        """
            This method returns the trained cnn_only model. If the model has already been
            trained, use the trained model instead. Else, trains the current cnn_only model
            and returns it.
        """
        
        if path.exists(CNN_ONLY_PATH):
            return load_model(CNN_ONLY_PATH, compile=True)
        else:
            print("[INFO] No cnn_only model have been trained.")
            self.train_cnn_only_model()
            return self.cnn_only_model
    
    
    