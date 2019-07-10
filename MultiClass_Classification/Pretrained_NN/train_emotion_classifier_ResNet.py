"""
Description: Train emotion classification model

Version using ResNet50 pretrained model as base_model
"""
import os
import keras
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras import layers
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.regularizers import l2
from keras.applications import resnet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.optimizers import Adam
from IPython.display import Image

dataset_path = 'FER_Project/MultiClass_Classification/Kaggle_Code/fer2013'
image_size=(48,48)
# parameters
batch_size = 128
num_epochs = 50
input_shape = (48, 48, 3)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'
l2_regularization=0.01

input_tensor = Input(input_shape)

#####
# Loading and Rescaling of datasets
#####
def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    #faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

#####
# Model definition
#####
resnet_model = resnet50.ResNet50(input_tensor=input_tensor, weights='imagenet',include_top=False)

x=resnet_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) # dense layers are added so that the model can learn more complex functions and classify for better results
x=Dense(1024,activation='relu')(x) # dense layer 2
x=Dense(512,activation='relu')(x) # dense layer 3
preds=Dense(7, activation='softmax')(x)

model = Model(resnet_model.input, preds)

# Training of the last dense layer
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[:20]:
    layer.trainable = False

for layer in model.layers[20:]:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#####
# Callbacks
#####
log_file_path = base_path + '_emotion_training_ResNet.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

#####
# Loading datasets and splitting into training (80 percent of dataset) and testing (20 percent of dataset)
#####
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
print("FACE LGTH = ",len(faces),"_ EMOTIONS LGTH = ",len(emotions))
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
