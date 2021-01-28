import re
import os
import cv2
import csv
import time
import h5py
import numpy as np
from utils import mkdir_p, BASE_PATH, CSV_PATH_DS3, IMG_PATH_DS3, HDF5_PATH_DS3, MOVED_FILES_PATH, errno, IMAGE_DIM
from datetime import datetime as dt
from backports.datetime_fromisoformat import MonkeyPatch
import matplotlib.pyplot as plt

MonkeyPatch.patch_fromisoformat()
time_steps = 15

def split_data(train_test_ratio):
	images = []
	labels = []
	insert_accel = 1
	insert_noaction = 2
	with open(os.path.join(CSV_PATH_DS3, 'training_data.csv'), 'r') as f:
		lines = f.readlines()
		reader = csv.reader(lines, delimiter=',')
		for line in reader:
			images.append(line[0].split()) #changes a list of strings to string list
			value = [(item) for item in line[1][1:-1].split(",")]
			value = [float(item) for item in value[:5]]
			value[1], value[2] = value[2], value[1]
			value[4], value[2] = value[2], value[4]
			if (value[0] < 0):
				value[insert_accel:insert_accel] = [1.0]
				value[0] = 0.0
			else:
				value[insert_accel:insert_accel] = [0.0]

			if value[0] == 0.0 and value[1] == 0.0:
				value[insert_noaction:insert_noaction] = [1.0]
			else:
				value[insert_noaction:insert_noaction] = [0.0]
			labels.append(value) 

	
	images = np.array(images) 
	labels = np.array(labels)
	print(f"Images shape: {images.shape}")
	print(f"Labels shape: {labels.shape}")
	
	
	data_size = images.shape[0]
	print('Total data size:', data_size)
	
	indices = np.arange(data_size)
	train_size = int(round(data_size * train_test_ratio))
	train_idx, test_idx = indices[:train_size], indices[train_size:]
	
	X_train = images[train_idx, :]
	Y_train = labels[train_idx, :]
	X_test = images[test_idx, :]
	Y_test = labels[test_idx, :]
	
	if X_train.shape[0] > 0:
		with open(os.path.join(HDF5_PATH_DS3, f'train_ts{time_steps}_ds3_224ktiny.txt'), 'w+') as f:
			for i in range(len(X_train)):
				f.write(f'{X_train[i][0]}|{list(Y_train[i][0:5])}\n')

	if X_test.shape[0] > 0:
		with open(os.path.join(HDF5_PATH_DS3, 'test.txt'), 'w+') as f:
			for i in range(len(X_test)):
				f.write(f'{X_test[i][0]}|{list(Y_test[i][:])}\n')

	
def collate_data(phase):

	with open(os.path.join(HDF5_PATH_DS3, f'{phase}_ts{time_steps}_ds3_224ktiny.txt'), 'r') as f:
		lines = f.readlines()
		data = csv.reader(lines, delimiter = "|")
	
	
	labels = []	
	collected_images =[]
	images_concat = []
	stamps = []	
	for index in data:
		
		img_id = index[0]
		stamps.append(img_id)
		for camera in ['centre','seg_c']:
			
			img_path = os.path.join(IMG_PATH_DS3, f'{camera}-{img_id}.jpg')
			image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			image = image[130:, :410]
			image = cv2.resize(image, IMAGE_DIM, interpolation=cv2.INTER_AREA)
		
			collected_images.append(image)
			
			
		concaImag = np.dstack(collected_images)
		images_concat.append(concaImag) #nx70x160x1
		
		collected_images = []
		concaImag = []
			
		label = label = [float(item) for item in index[1][1:-1].split(",")]	
		labels.append(label)
		
	
	print("length of images_concat", len(images_concat))
	print("length of labels", len(labels))
	print("length of stamps", len(stamps))
	
	make_data_time_series(stamps, images_concat, labels, phase)

def make_data_time_series(stamps, ic, labels, phase):

	print(f"Making data time series started at {time.ctime()}")
	dataX1, dataX2, dataY = [], [], []
	frames_skipped = 0
	#check this condition
	#if len(ic) - time_steps <= 0:
	#	dataX.append(ic)        
	#else:
	for i in range(len(stamps) - time_steps):
		
		diff = dt.fromisoformat(stamps[i+1]) - dt.fromisoformat(stamps[i])
		if (diff.total_seconds() < 1):
			a1 = ic[i:(i + time_steps)]
			a2 = labels[i:(i + time_steps)]
			dataX1.append(a1)
			dataX2.append(a2)
			dataY.append(labels[i + (time_steps-1)]) 
		else:
			frames_skipped +=1
	ts_images = np.array(dataX1)
	ts_labels1 = np.array(dataX2)
	ts_labels = np.array(dataY)
	
	print(f"Number of frames skipped: {frames_skipped}")
	print(f"ts_images final shape {ts_images.shape}")
	print(f"ts_labels1 final shape {ts_labels1.shape}")
	print(f"ts_labels final shape {ts_labels.shape}")

	print(f"Making data time series ended at {time.ctime()}")
	print("Writing to hdf5....")
	write_to_hdf5(ts_images, ts_labels,ts_labels1, phase)	

def write_to_hdf5(X_data, Y_data, X_data1, phase):
	h5_file = os.path.join(HDF5_PATH_DS3, f'{phase}_ts{time_steps}_ds3_224ktiny.h5')
	h5_file1 = os.path.join(HDF5_PATH_DS3, f'{phase}_ts{time_steps}_ds3_224ktinya.h5')
			
	with h5py.File(h5_file, 'w') as f:
		f.create_dataset('data', data=X_data)
		f.create_dataset('label', data=Y_data)
	
	with open(os.path.join(HDF5_PATH_DS3, f'{phase}_ts{time_steps}_ds3_224ktiny_h5_list.txt'), 'a+') as f:
		f.write(h5_file + '\n')

	with h5py.File(h5_file1, 'w') as f:
		f.create_dataset('data', data=X_data1)
		f.create_dataset('label', data=Y_data)
	
	with open(os.path.join(HDF5_PATH_DS3, f'{phase}_ts{time_steps}_ds3_224ktinya_h5_list.txt'), 'a+') as f:
		f.write(h5_file1 + '\n')
			
def truncate_hdf5():
	for f in [f for f in os.listdir(HDF5_PATH_DS3)]:
		os.remove(os.path.join(HDF5_PATH_DS3, f))

if __name__ == '__main__':
	t0 = time.time()
	print(f"Preprocessing started at {time.ctime()}")
	mkdir_p(HDF5_PATH_DS3)
	#truncate_hdf5()
	split_data(1.0) #Since sklearn is used in training module
	print(f"Splitting the data finished at {time.ctime()}")
	collate_data('train')
	print(f"Preprocessing finished at {time.ctime()}")
	#prepare_write_to_hdf5('test')
	t1 = time.time()
	print('Total elapsed time:', t1 - t0, 'seconds')

'''
'''
