"""
Description: Train emotion classification model
"""
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras import optimizers
import pandas as pd
import cv2
import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf

dataset_path = 'fer2013/fer2013.csv'
image_size=(48,48)
# Parameters
batch_size = 64
num_epochs = 100
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 6
base_path = 'models/'
l2_regularization=0.01
learning_rate = 0.001

""" Script to load JAFFE Dataset as a training dataset
def load_jaffeDS():
    #dir_list = os.listdir(dataset_path)
    #for(dir in dir_list):
    faces = []
    emotion_list = []
    i=0
    for path, dirs, files in os.walk(dataset_path):
        for filename in files:
            if(filename!='.DS_Store'):
                current_dir = os.path.basename(path)
                print("filename = ", filename, "_dir = ", current_dir)
                origimg = cv2.imread(path+"/"+filename,0)
                print(origimg)
                height, width = origimg.shape
                newimg = cv2.resize(origimg,(48,48))
                cv2.imwrite(filename,newimg)
                faces.append(newimg.astype('float32'))
                emotion_list.append(getRankedEmotionNumber(current_dir))
        i+=1
    emotions = pd.get_dummies(emotion_list).as_matrix()
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    return faces, emotions

def getRankedEmotionNumber(dir):
    if(dir=="ANG"):
        return 0
    if(dir=="DIS"):
        return 1
    if(dir=="FEA"):
        return 2
    if(dir=="HAP"):
        return 3
    if(dir=="SAD"):
        return 4
    if(dir=="SUR"):
        return 5
    if(dir=="NEU"):
        return 6
"""

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
		faces.append(face.astype('float32'))
	faces = np.asarray(faces)
	faces = np.expand_dims(faces, -1)
	emotions = pd.get_dummies(data['emotion']).as_matrix()
	return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

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
# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
regularization = l2(l2_regularization)

# base
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# module 1
residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 2
residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 3
residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 4
residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])
x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax',name='predictions')(x)

adam=optimizers.Adam(lr=learning_rate)

model = Model(img_input, output)
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#####
# Callbacks
#####
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, mode='auto')
#reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.8, patience=3, verbose=1)
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
tensor_board = TensorBoard(log_dir='/home/julie/Lucine_MLTest/Kaggle_Test/FaceEmotion_ID-master/logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr,tensor_board]

#####
# Loading datasets and splitting into training (80 percent of dataset) and testing (20 percent of dataset)
#####
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))

#####
# Model evaluation using standardized dataset
#####
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=model, epochs=100, batch_size=128, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)

predicted_test_labels = np.argmax(model.predict(xtest), axis=1)
test_labels = np.argmax(ytest, axis=1)

score, accuracy =  model.evaluate(xtest, ytest, batch_size=batch_size);
print("Accuracy = ",accuracy,"_Score = ",score)
