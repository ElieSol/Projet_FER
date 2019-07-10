import pandas as pd
import numpy as np
import os
import sys
import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.models import Model
from keras.optimizers import Adam
from IPython.display import Image
from IPython.display import display

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


#####
# Model definition
#####
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1, 1), input_shape=(224,224,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=2, activation='sigmoid'))

model.summary()


#####
# Loading and Rescaling of training and test datasets
#####
# Image Augmentation was used to avoir overfitting due to the small amount of images in both datasets
train_datagen = ImageDataGenerator(
 rescale=1./255, # Rescales the value of the pixels images from [0,255] to 0 or 1 values
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator=train_datagen.flow_from_directory('/Users/Juliesolacroup/Documents/GitHub/FER_Project/FER_Project/Datasets/FEI/Training', target_size=(224,224), color_mode='rgb', batch_size=25, class_mode='categorical', shuffle=True)

test_set = test_datagen.flow_from_directory('/Users/Juliesolacroup/Documents/GitHub/FER_Project/FER_Project/Datasets/FEI/Test', target_size=(224,224), color_mode='rgb', batch_size=25)

#####
# Compilation of the model
#####
# Adam is used as the optimizer
# Binary_crossentropy is used as the loss
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# other optimizer: adadelta

# Equations used as reference for the computation of the number of steps per epoch and the number of validation_steps:
#   steps_per_epoch = Total Training Samples/Training Batch Size
#   validation_steps = Total Validation Samples / Validation Batch Size
validation_steps_nb = test_set.n//test_set.batch_size
step_size_train = train_generator.n//train_generator.batch_size
model.fit_generator(train_generator, steps_per_epoch=step_size_train, epochs=8, validation_data=test_set, validation_steps=validation_steps_nb)



#####
# Testing of the model with a given image
#####
def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    return img_tensor

img_path = sys.argv[0]

train_generator.class_indices
print(train_generator.class_indices)

# Predicting the test image
new_image = load_image(img_path)
pred = model.predict(new_image)
