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
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import load_model
from datetime import datetime
from os import path
from keras.models import Model

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
        self.cnn_landmarks_model = self.__build_cnn_landmarks_model()
    
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
            MaxPooling2D(pool_size=(2,2), strides=2),
            #
            Conv2D(128,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
            #
            Conv2D(512,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
            #
            Conv2D(512,(3,3), activation='relu'), 
            MaxPooling2D(pool_size=(2,2), strides=2),
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
    
    def get_cnn_landmarks_model(self):
        self.cnn_landmarks_model.compile(loss=LOSS, optimizer="adam", metrics=["accuracy"])
        return self.cnn_landmarks_model
    
    def get_conv_layers(self, input_shape = INPUT_SHAPE):
        model_in = Input(shape=input_shape, name='ConvInput')
        
        conv1 = Conv2D(64,(5,5), activation='relu')(model_in)
        pooling1 = MaxPooling2D(pool_size=(2,2), strides=2)(conv1)
        
        conv2 = Conv2D(128,(3,3), activation='relu')(pooling1)
        pooling2 = MaxPooling2D(pool_size=(2,2), strides=2)(conv2)
        
        conv3 = Conv2D(512,(3,3), activation='relu')(pooling2)
        pooling3 = MaxPooling2D(pool_size=(2,2), strides=2)(conv3)
        
        conv4 = Conv2D(512,(3,3), activation='relu')(pooling3)
        pooling4 = MaxPooling2D(pool_size=(2,2), strides=2)(conv4)
        
        flat1 = Flatten()(pooling4)
        model_out = Dense(128, activation='relu')(flat1)
        
        return model_in, model_out
    
    def get_landmarks_layers(self, input_shape=(68,2)):
        
        model_in = Input(shape=input_shape, name='LandmarksInput')
        
        dense1 = Dense(68, activation='relu')(model_in)
        flat1 = Flatten()(dense1)
        
        model_out = Dense(128, activation='relu') (flat1)
        
        return model_in, model_out
    
    
    def __build_cnn_landmarks_model(self):
        
        """
            This method build a non-linear network model for the CNN + landmarks approach. 
            The model takes has inputs a normalized grayscaled image resized to 48x48
            pixels dimensions and the 68 facials landmarks. the output is a 7 emotions 
            classes probability vector obtained after classification.
        """ 
        
        # Getting the inputs and outputs layers of the two branches
        conv_model_in, conv_model_out = self.get_conv_layers()
        landmarks_model_in, landmarks_model_out = self.get_landmarks_layers()
        
        # Merging the branches together
        merged_model1 = Concatenate()([landmarks_model_out, conv_model_out])
        
        # Flattening the merged model
        flat_merged = Flatten()(merged_model1)
        
        # 256 & 512 fully connected layers 
        dense1 = Dense(256, activation='relu')(flat_merged)
        dense2 = Dense(512, activation='relu')(dense1)
        
        # Model output classified into 7 emotions
        model_out = Dense(7, activation='softmax')(dense2)
        
        cnn_landmarks_model = Model(inputs=[conv_model_in, landmarks_model_in], outputs=[model_out])
        
        return cnn_landmarks_model
    
    