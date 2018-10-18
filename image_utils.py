# -*- encoding=UTF-8-*-

import os
import gc
import numpy as np
from osgeo import gdal
import random
import h5py
from skimage.transform import resize, rotate
from sklearn import decomposition
from skimage.filters import sobel, rank
from skimage.morphology import disk
from tqdm import tqdm
import psutil
from random import randint

def load_file(path, resizeTo=None):
	dataSource = gdal.Open(path)

	if (dataSource is not None):
		bands = []
		for index in range(1, dataSource.RasterCount + 1):
			band = dataSource.GetRasterBand(index).ReadAsArray()

			if (resizeTo):
				band = resize(band, resizeTo, preserve_range=True, anti_aliasing=True).astype(np.int16)

			bands.append(band)

		image = np.dstack(bands)

		return image
	else:
		return None

def normalize(image):
	mean = np.mean(image)
	std = np.std(image)
	normalized = ((image - mean) / float(std))
	return  normalized

def get_gradient(image):
	# aplicacao do PCA
	num_bands = image.shape[2]
	
	bands = []
	for i in range(num_bands):
		band = image[:, :, i:i+1]
		band = rank.median(band.reshape(band.shape[0], band.shape[1]), disk(1))
		bands.append(band)
	
	image = np.dstack(bands)
	
	image_flat = image.reshape(-1, num_bands)
	pca = decomposition.PCA(n_components=num_bands)
	pca.fit(image_flat)

	image_flat_pca = pca.transform(image_flat)
	image_pca = image_flat_pca.reshape(image.shape)
	image_pca = image_pca.transpose((-1, 0, 1))

	# geração dos gradientes
	gradients = []
	for band in image_pca[:3]:
		gradients.append(sobel(band))

	# MAX dos gradientes
	gradient = np.sum(gradients, axis=0)

	return gradient


def get_rotate(image):
	images = []
	for rot in [0, 90, 180, 270]:
		image_rotate = rotate(image, rot, preserve_range=True)
		images.append(image_rotate)
	return images

def get_flip(image):
	horizontal_flip = image[:, ::-1]
	vertical_flip = image[::-1, :]
	return [horizontal_flip, vertical_flip]

def load_dataset(dataset, read_only=False):
	if(read_only):
		dataset = h5py.File(dataset, 'r')
	else: 
		dataset = h5py.File(dataset, 'r+')
	x_data = dataset["x"]
	y_data = dataset["y"]
	return dataset, x_data, y_data

def make_dataset(filename, width, height, channels):
	dataset = h5py.File(filename, 'w')
	x_data = dataset.create_dataset("x", (0, width, height, channels), 'f', maxshape=(None, width, height, channels), chunks=True)
	y_data = dataset.create_dataset("y", (0, width, height, 1), 'f', maxshape=(None, width, height, 1), chunks=True)
	return dataset, x_data, y_data

def get_available_memory():
	return 100 - psutil.virtual_memory().percent	

def generate_dataset(image_path, labels_path, output_path, chip_size, channels, grids=1, rotate=False, flip=False):
	image_data = load_file(image_path)
	image_labels = load_file(labels_path)

	# resize labels
	image_labels = resize(image_labels, (image_data.shape[0], image_data.shape[1]), preserve_range=True, anti_aliasing=True).astype(np.int8)

	image = np.dstack([image_data, image_labels])
	del image_data
	del image_labels
	
	for step in get_grids(grids, chip_size):
		batch = []
		for (x, y, window, dimension) in sliding_window(image, step["steps"], step["chip_size"], (chip_size, chip_size)):
			if get_available_memory() >= 10:
				train = np.array(window[:, :, : channels], dtype=np.int16)
				labels = np.array(window[:, :, -1:], dtype=np.int8)

				train = normalize(train)
				labels_unique = np.unique(labels)

				if not (0 in labels_unique and len(labels_unique) == 1):
					image_raw = np.dstack([train, labels])
					images_daugmentation = []

				
					if (rotate):
						images_rotate = get_rotate(image_raw)
						images_daugmentation.extend(images_rotate)
					elif (flip):
						images_flip = get_flip(image_raw)
						images_daugmentation.extend(images_flip)
					else:
						images_daugmentation.append(image_raw)

					for i in images_daugmentation:
						new_train = i[:, :, : channels]
						new_labels = np.array(i[:, :, -1:], dtype=np.int8)
						batch.append((new_train, new_labels))
		
			else:
				save_dataset(batch, output_path, chip_size, channels)
				del batch
				batch = []

		save_dataset(batch, output_path, chip_size, channels)
		del batch
		batch = []

def save_dataset(batch, output_path, chip_size, channels):
	if os.path.isfile(output_path):
		dataset, x_data, y_data = load_dataset(output_path)
	else:
		dataset, x_data, y_data = make_dataset(output_path, chip_size, chip_size, channels)

	length = len(batch)

	x_data_size = x_data.len()
	y_data_size = y_data.len()

	x_data.resize((x_data_size + length, chip_size, chip_size, channels))
	y_data.resize((y_data_size + length, chip_size, chip_size, 1))

	for index in tqdm(iterable=range(length), miniters=10, unit=" samples"):
		x_data[x_data_size + index] = batch[index][0]
		y_data[y_data_size + index] = batch[index][1]

	dataset.close()

def sliding_window(image, step, windowSize, windowResize=None):
	# slide a window across the image
	for y in range(0, image.shape[0], int(step[1])):
		for x in range(0, image.shape[1], int(step[0])):
			# yield the current window
			window = image[ y:y + windowSize[1], x:x + windowSize[0]]
			original_shape = window.shape

			if (windowResize != None):
				window = resize(window, (windowResize[0], windowResize[1]), preserve_range=True, anti_aliasing=True).astype(np.int16)
			yield (x, y, window, original_shape)


def get_window(matrix, x, y, width, height):
	return matrix[ y:y + height, x:x + width]

def set_window(matrixA, matrixB, x, y):
	for i_index, i in enumerate(range(y, y + matrixB.shape[0])):
		for j_index, j in enumerate(range(x, x + matrixB.shape[1])):
			try:
				matrixA[i][j] = matrixB[i_index][j_index]
			except Exception as e:
				import ipdb; ipdb.set_trace()


def get_grids(grids, chip_size):
	grids_dict = {
		1: [
			{"steps":  (chip_size, chip_size), "chip_size":  (chip_size, chip_size)},
		],
		2 : [
			
			{"steps":  (chip_size * 0.5, chip_size * 0.5), "chip_size":  (chip_size, chip_size)},              
        	],
		3 : [ 
               
                	{"steps":  (chip_size * 0.5, chip_size * 0.5), "chip_size":  (chip_size, chip_size)},
                	{"steps":  (chip_size * 0.5, chip_size * 0.5), "chip_size":  (chip_size * 2, chip_size * 2)},
        	]
	}
	
	return grids_dict[grids]
