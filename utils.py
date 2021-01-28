import h5py
import numpy as np
import math
import cv2
import os
import errno

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CSV_PATH_DS1 = u'{}/data/ds1'.format(BASE_PATH)
CSV_PATH_DS2 = u'{}/data/ds2'.format(BASE_PATH)
CSV_PATH_DS3 = u'{}/data/ds3'.format(BASE_PATH)
IMG_PATH_DS1 = u'{}/data/ds1/img'.format(BASE_PATH)
IMG_PATH_DS2 = u'{}/data/ds2/img'.format(BASE_PATH)
IMG_PATH_DS3 = u'{}/data/ds3/img'.format(BASE_PATH)
HDF5_PATH_DS1 = u'{}/data/hdf5/ds1'.format(BASE_PATH)
HDF5_PATH_DS2 = u'{}/data/hdf5/ds2'.format(BASE_PATH)
HDF5_PATH_DS3 = u'{}/data/hdf5/ds3'.format(BASE_PATH)
RADAR_PATH = u'{}/data/radar'.format(BASE_PATH)
MODEL_PATH = u'{}/model'.format(BASE_PATH)
MOVED_FILES_PATH = u'{}/data/handbrake_reverse_images'.format(BASE_PATH)
IMAGE_DIM = (160, 70) #changed from (320, 70)


def load_dataset(file_name):
	hdf5_path = '{}'.format(file_name)
	hdf5_file = h5py.File(hdf5_path, 'r')

	data_X = np.array(hdf5_file['data'][:])
	data_Y = np.array(hdf5_file['label'][:])

	return data_X, data_Y


def load_multi_dataset(txt_name):
	data_X, data_Y = None, None
	try:
		with open(txt_name, 'r') as f:
			for line in f:
				line = line.rstrip()
				x, y = load_dataset(line)
				if data_X is None or data_Y is None:
					data_X, data_Y = x, y
				else:
					data_X = np.concatenate((data_X, x))
					data_Y = np.concatenate((data_Y, y))
	except FileNotFoundError as e:
		print(e)
		return data_X, data_Y

	return data_X, data_Y

			

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
	m = X.shape[0]
	mini_batches = []
	np.random.seed(seed)
	
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	num_complete_minibatches = int(math.floor(m/mini_batch_size))
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches
 

def mkdir_p(path):
	try:
		os.makedirs(path)
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise
	print('Directory is ready: {}'.format(path))


if __name__ == '__main__':
	load_multi_dataset('hdf5/train_h5_list.txt')
