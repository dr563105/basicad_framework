import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import MaxPooling2D, LSTM, TimeDistributed, concatenate, BatchNormalization, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
#import tensorflow.keras.backend as K
from keras.utils import plot_model, to_categorical
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from utils import load_multi_dataset, mkdir_p, HDF5_PATH_DS3, MODEL_PATH
from datetime import datetime
import time
import errno
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')

from config import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID_TO_USE

#Loading the model, splitting it into train and test sets using Sklearn library.
print(f'Loading data from HDF5... at {time.ctime()}')

X_data, Y_data = load_multi_dataset(os.path.join(HDF5_PATH_DS3, f'train_ts{TIMESTEPS}_ds3_3_h5_list.txt'))

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

input_normalisation= Lambda(lambda x: x / 255.0) (input_layer)

#Conv layers
conv2d_layer_1 = TimeDistributed(Conv2D(CONV2D_FILTERS_1, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (input_normalisation)
conv2d_layer_2 = TimeDistributed(Conv2D(CONV2D_FILTERS_2, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_1)
conv2d_layer_3 = TimeDistributed(Conv2D(CONV2D_FILTERS_3, KERNEL_SIZE_1, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_2)
conv2d_layer_4 = TimeDistributed(Conv2D(CONV2D_FILTERS_4, KERNEL_SIZE_2, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_3)
conv2d_layer_5 = TimeDistributed(Conv2D(CONV2D_FILTERS_4, KERNEL_SIZE_2, strides=STRIDE_DIM_1, padding=PADDING, activation=CONV2D_ACTIVATION_FN)) (conv2d_layer_4)
flatten_layer = TimeDistributed(Flatten()) (conv2d_layer_5)

#LSTM layers
lstm_layer = LSTM(LSTM_OUTPUT_UNITS, activation=LSTM_ACTIVATION_FN) (flatten_layer)

#Dense layers
dense_layer_1 = Dense(DENSE_HIDDEN_UNITS_1, activation=DENSE_ACTIVATION_FN) (lstm_layer)

dense_layer_2 = Dense(DENSE_HIDDEN_UNITS_2, activation=DENSE_ACTIVATION_FN) (dense_layer_1)
dense_layer_3 = Dense(DENSE_HIDDEN_UNITS_3, activation=DENSE_ACTIVATION_FN) (dense_layer_2)
dense_layer_4 = Dense(DENSE_HIDDEN_UNITS_3, activation=DENSE_ACTIVATION_FN) (dense_layer_2)

#Output layers
output_steering = Dense(1, activation = DENSE_OUTPUT_ACTIVATION_FN_STEERING, name='st') (dense_layer_3)
output_accel = Dense(1, activation = DENSE_OUTPUT_ACTIVATION_FN_ACCELERATION, name='accel') (dense_layer_4)

output_combo = concatenate([output_accel, output_steering], name = 'acc_st')

#Model definition
model = Model(inputs=input_layer, outputs=output_combo , name='evaluate_ts15_ds3_tanh1')

model.summary()
plot_model(model, to_file=PLOT_MODEL_SAVE_FILE, show_shapes=PLOT_MODEL_SHOW_SHAPES)
model.compile(optimizer=Adam(lr=MODEL_LEARNING_RATE, decay = MODEL_LEARNING_DECAY), loss={'acc_st': MODEL_LOSS_FN1})

#print("Loading model_criteria68....")
#model = load_model(os.path.join(MODEL_PATH, 'model_criteria68.h5'))

#Callbacks definition
es = EarlyStopping(monitor=CALLBACKS_MONITOR, mode=CALLBACKS_MONITOR_MODE, verbose=CALLBACKS_VERBOSITY, patience=EARLYSTOPPING_PATIENCE)
mc = ModelCheckpoint(os.path.join(MODEL_PATH, MODEL_CHECKPOINT_FILENAME),
	monitor=CALLBACKS_MONITOR, mode=CALLBACKS_MONITOR_MODE, verbose=CALLBACKS_VERBOSITY, save_best_only=MODEL_CHECKPOINT_SAVE_BEST)
logdir = TENSORBOARD_LOG_PATH
tbc = TensorBoard(log_dir=logdir)

#Fitting the model
t0 = time.time()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=MODEL_FIT_SHUFFLE, epochs=TRAINING_EPOCH, batch_size=BATCH_SIZE,
    verbose=MODEL_FIT_VERBOSITY, callbacks=[mc, tbc])

t1 = time.time()
print('Total training time:', t1 - t0, 'seconds')
