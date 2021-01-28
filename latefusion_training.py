import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import MaxPooling2D, LSTM, TimeDistributed, concatenate, BatchNormalization, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.utils import plot_model, to_categorical
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from utils import load_multi_dataset, mkdir_p, HDF5_PATH_DS3, MODEL_PATH
from datetime import datetime
import time
import errno
from sklearn.model_selection import train_test_split
import functools
from config import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID_TO_USE

#Custom weighting functions
def w_categorical_crossentropy(y_true, y_pred, sample_weight=SAMPLE_WEIGHT):
	#x=tf.keras.losses.CategoricalCrossentropy(y_pred, y_true)
	
	cce = tf.keras.losses.CategoricalCrossentropy()
	x=cce(y_true, y_pred)
	y=tf.reduce_mean(x)
	#print("Shape of cce",x)
	#print("Shape of cce after mean",y)
	return y*WEIGHT_FOR_CCE 


def w_mse(y_true, y_pred, sample_weight=SAMPLE_WEIGHT):
	x=tf.square(y_true - y_pred)
	y=tf.reduce_mean(x)
	#print("Shape of mse",x)
	#print("Shape of mse after mean ",y)
	return y*WEIGHT_FOR_MSE 

def st_mse(y_true, y_pred, sample_weight=SAMPLE_WEIGHT):
	x=tf.square(y_true - y_pred)
	y=tf.reduce_mean(x)
	#print("Shape of st_mse",x)
	#print("Shape of st_mse after mean ",y)
	return y*WEIGHT_FOR_ST

ncce = functools.partial(w_categorical_crossentropy,sample_weight=SAMPLE_WEIGHT)
ncce.__name__ = 'cce'

nmse = functools.partial(w_mse,sample_weight=SAMPLE_WEIGHT)
nmse.__name__ = 'mse'

nstmse = functools.partial(st_mse,sample_weight=SAMPLE_WEIGHT)
nstmse.__name__ = 'stmse'

#Loading the model, splitting it into train and test sets using Sklearn library.
print(f'Loading data from HDF5... at {time.ctime()}')

X_data, Y_data = load_multi_dataset(os.path.join(HDF5_PATH_DS3, f'train_ts{TIMESTEPS}_ds3_7_h5_list.txt'))

print('Number of images:', X_data.shape)
print('Number of labels:', Y_data.shape)

print(f'Splitting data into training set and testing set....at {time.ctime()}')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=SKLEARN_TRAIN_TEST_SPLIT_SIZE, random_state=SKLEARN_RANDOM_STATE)
print(f'Splitting data into training set and testing end....at {time.ctime()}')

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

input_layer = Input(shape = TIMESTEPS_INPUT_SHAPE)
#Prepare for late fusion and split last tensor
input_1 = Lambda(lambda x:x[ :, :, :, :, 0:1]) (input_layer) #splitting the last channel axis into two
input_2 = Lambda(lambda x:x[ :, :, :, :, 1:2]) (input_layer)

#Take the first one and do convolutional operation
input_normalisation_1= Lambda(lambda x: x / 255.0) (input_1)
conv2d_layer_1 = TimeDistributed(Conv2D(CONV2D_FILTERS_1, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (input_normalisation_1) 
conv2d_layer_2 = TimeDistributed(Conv2D(CONV2D_FILTERS_2, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_1) 

#Take the second one and do convolutional operation
input_normalisation_2= Lambda(lambda x: x / 255.0) (input_2)
conv2d_layer_11 = TimeDistributed(Conv2D(CONV2D_FILTERS_1, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (input_normalisation_2) 
conv2d_layer_22 = TimeDistributed(Conv2D(CONV2D_FILTERS_2, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_11) 

#Concatenate the two conv-ed feature maps
concatenate_layer = concatenate([conv2d_layer_2, conv2d_layer_22])

#Continue conv operations
conv2d_layer_3 = TimeDistributed(Conv2D(CONV2D_FILTERS_5, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (concatenate_layer)
conv2d_layer_4 = TimeDistributed(Conv2D(CONV2D_FILTERS_6, KERNEL_SIZE_2, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_3)
conv2d_layer_5 = TimeDistributed(Conv2D(CONV2D_FILTERS_6, KERNEL_SIZE_2, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_4) 

#Flatten into a single feature vector
flatten_layer = TimeDistributed(Flatten()) (conv2d_layer_5)

#lstm_layer_common = LSTM(200, activation='tanh',return_sequences=True) (flatten_layer)

#LSTM Layers
lstm_layer = LSTM(LSTM_OUTPUT_UNITS, activation=LSTM_ACTIVATION_FN) (flatten_layer)
lstm_layer2 = LSTM(LSTM_OUTPUT_UNITS, activation=LSTM_ACTIVATION_FN) (flatten_layer)
lstm_layer3 = LSTM(LSTM_OUTPUT_UNITS, activation=LSTM_ACTIVATION_FN) (flatten_layer)

#Dense layers
dense_layer_1 = Dense(DENSE_HIDDEN_UNITS_1, activation=DENSE_ACTIVATION_FN) (lstm_layer)
dense_layer_2 = Dense(DENSE_HIDDEN_UNITS_1, activation=DENSE_ACTIVATION_FN) (lstm_layer2)
dense_layer_3 = Dense(DENSE_HIDDEN_UNITS_1, activation=DENSE_ACTIVATION_FN) (lstm_layer3)

dense_layer_4 = Dense(DENSE_HIDDEN_UNITS_3, activation=DENSE_ACTIVATION_FN) (dense_layer_1)
dense_layer_5 = Dense(DENSE_HIDDEN_UNITS_3, activation=DENSE_ACTIVATION_FN) (dense_layer_2)
dense_layer_6 = Dense(DENSE_HIDDEN_UNITS_3, activation=DENSE_ACTIVATION_FN) (dense_layer_3)

#Output layers
output_steering = Dense(1, activation = DENSE_OUTPUT_ACTIVATION_FN_STEERING, name='st') (dense_layer_6)
output_velocity = Dense(1, activation = DENSE_OUTPUT_ACTIVATION_FN_VELOCITY, name='velocity') (dense_layer_5)
#choose from accelerate, brake, or no action states
output_classification = Dense(3, activation = DENSE_OUTPUT_ACTIVATION_FN_CLASSIFICATION, name='classification') (dense_layer_4)

#Model definition
model = Model(inputs=input_layer, outputs=[output_classification, output_velocity, output_steering] , name='evaluate_ts15_ds3_Class6_cseg2')

model.summary()
plot_model(model, to_file=PLOT_MODEL_SAVE_FILE, show_shapes=PLOT_MODEL_SHOW_SHAPES)

model.compile(optimizer=MODEL_OPTIMIZER(lr=MODEL_LEARNING_RATE, decay = MODEL_LEARNING_DECAY), loss={'classification':ncce, 'st':nstmse,'velocity':nmse})


#print("Loading model_criteria68....")
#model = load_model(os.path.join(MODEL_PATH, 'evaluate_ts15_ds3_Class4_6.h5'))

#Callbacks definitions
es = EarlyStopping(monitor=CALLBACKS_MONITOR, mode=CALLBACKS_MONITOR_MODE, verbose=CALLBACKS_VERBOSITY, patience=EARLYSTOPPING_PATIENCE)
mc = ModelCheckpoint(os.path.join(MODEL_PATH, MODEL_CHECKPOINT_FILENAME), 
	monitor=CALLBACKS_MONITOR, mode=CALLBACKS_MONITOR_MODE, verbose=CALLBACKS_VERBOSITY, save_best_only=MODEL_CHECKPOINT_SAVE_BEST)
logdir = TENSORBOARD_LOG_PATH
tbc = TensorBoard(log_dir=logdir)

#Fitting the model
t0 = time.time()
model.fit(x=[X_train, Y_train], {'classification': Y_train,'st': Y_train[:,-2],'velocity': Y_train[:,-1] } , 
	validation_data=(x=[X_test, Y_test[:,-1]], {'classification': Y_test[:,0:3],'st': Y_test[:,-2],'velocity': Y_test[:,-1]}), 
	shuffle=MODEL_FIT_SHUFFLE, epochs=TRAINING_EPOCH, batch_size=BATCH_SIZE, 
	verbose=MODEL_FIT_VERBOSITY, callbacks=[mc, tbc])

t1 = time.time()
print('Total training time:', t1 - t0, 'seconds')