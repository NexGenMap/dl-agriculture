import os
import sys
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import tensorflow as tf
from osgeo import gdal
import psutil
import shutil
import glob

from models import unet, deeplab
import image_utils
import arguments

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 1

models = {
	'unet'      : unet.model_fn,
	'deeplab'   : deeplab.model_fn,
}

params={
	
	'tensorboard_images_max_outputs': 2,
	'learning_rate': 0.00005,
}

def train(input_train, input_test, model, model_dir, epochs, batch_size, params):
	train_file, train_data, train_labels = image_utils.load_dataset(input_train, read_only=True)
	test_file, test_data, test_labels = image_utils.load_dataset(input_test, read_only=True)
	
	train_size = train_data.len()
	test_size  = test_data.len()

	print("\nTrain: {0}\nTest:{1}\n".format(train_size, test_size))

	logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=train_size)

	if model not in models.keys():
		raise ValueError('Model ' + model + ' is not supported')

	model_fn=models[model]

	with tf.Session(config=config) as sess:
		FILE_BATCH  = 20000
		train_input = None
		test_input  = None
		lowest_loss = 1
	
		estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_dir)

		for epoch in range(1, epochs + 1):		
			print("\nEPOCH: {index}\n".format(index=epoch))
		
			train_index = 0
			while train_index < train_size:
				if FILE_BATCH == None:
					if train_input == None:
						train_input = tf.estimator.inputs.numpy_input_fn(
							x={"data": np.asarray(train_data[train_index:], dtype=np.float32)},
							y=np.asarray(train_labels[train_index:], dtype=np.int8),
							batch_size= batch_size,
							num_epochs=1,
							shuffle=True)
					train_index = train_size
				else:
					index = train_index % train_size	
					train_input = tf.estimator.inputs.numpy_input_fn(
						x={"data": np.asarray(train_data[index:index + FILE_BATCH], dtype=np.float32)},
						y=np.asarray(train_labels[index:index + FILE_BATCH], dtype=np.int8),
						batch_size= batch_size, 
						num_epochs=1,
						shuffle=True)
					train_index += FILE_BATCH

				train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])
				if not FILE_BATCH == None:
					del train_input
				#break
			
	
			print(train_results)
		
			test_index = 0
			while test_index < test_size:
				if FILE_BATCH == None:
					if test_input == None:
						test_input = tf.estimator.inputs.numpy_input_fn(
							x={"data": np.asarray(test_data[test_index:], dtype=np.float32)},
							y=np.asarray(test_labels[test_index:], dtype=np.int8),
							num_epochs=1,
							batch_size=batch_size,
							shuffle=True
						)
					test_index = test_size
				else:
					index = test_index % test_size
					test_input = tf.estimator.inputs.numpy_input_fn(
						x={"data": np.asarray(test_data[index:index+FILE_BATCH], dtype=np.float32)},
						y=np.asarray(test_labels[index:index+FILE_BATCH], dtype=np.int8),
						num_epochs=1,
						batch_size=batch_size,
						shuffle=False)
					test_index += FILE_BATCH

				test_results = estimator.evaluate(input_fn=test_input, steps=None, hooks=[logging_hook])
				if not FILE_BATCH == None:
					del test_input
				#break

			print(test_results)
		
			# save best checkpoint
			current_loss = test_results.get("loss")
			if current_loss < lowest_loss:
				lowest_loss = current_loss
				best_model_dir = "{model_dir}/best".format(model_dir=model_dir)
				if not os.path.exists(best_model_dir):
					os.makedirs(best_model_dir)
				latest_checkpoint = estimator.latest_checkpoint()
				shutil.copy(model_dir+"/checkpoint", best_model_dir)
				for f in glob.glob(latest_checkpoint+"*"):
					shutil.copy(f, best_model_dir)
		

		sess.close()	

def evaluate(input_validation, model, model_dir, batch_size, params):
	validation_file, validation_data, validation_labels = image_utils.load_dataset(input_validation, read_only=True)
	print(np.unique(validation_labels))
	params['batch_size'] = batch_size

	if model not in models.keys():
		raise ValueError('Model ' + model + ' is not supported')

	model_fn=models[model]

	with tf.Session(config=config) as sess:	
		sess.run(tf.global_variables_initializer())
	
		estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_dir)

		validation_input = tf.estimator.inputs.numpy_input_fn(x={"data": validation_data}, batch_size=batch_size, shuffle=False)
		validation_results = estimator.predict(input_fn=validation_input)
		
		mean_acc = []

		i = 0

		for predict, expect in zip(validation_results, validation_labels):
		
			predict[predict >= 0.5]  = 1
			predict[predict < 0.5] = 0

			expect[expect >= 0.5]  = 1
			expect[expect < 0.5]  = 0
			
			
			pre_flat = predict.reshape(-1)
			exp_flat = expect.reshape(-1)
			
			mean_acc.append(accuracy_score(exp_flat, pre_flat))
			i = i + 1

		print('\n--------------------------------------------------')
		print('---------------------- METRICS ---------------------')
		print('----------------------------------------------------')
		print('Validation accurancy:',np.mean(mean_acc))
		print(classification_report(exp_flat, pre_flat))

def predict(input_path, output_path, model, model_dir, chip_size, channels, grids, batch_size, params):
	
	input_dataset = gdal.Open(input_path)

	image = image_utils.load_file(input_path)[:, : , :channels]

	image_predicted = np.zeros((image.shape[0], image.shape[1]), dtype=np.int)
	
	params['batch_size'] = batch_size

	if model not in models.keys():
		raise ValueError('Model ' + model + ' is not supported')

	model_fn=models[model]

	estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_dir)

	for step in image_utils.get_grids(grids, chip_size):
		batch = []
		for (x, y, window, original_dimensions) in image_utils.sliding_window(image, step["steps"], step["chip_size"], (chip_size, chip_size)):
			if window.shape[0] != chip_size or window.shape[1] != chip_size:
				print(window.shape, chip_size)
				continue

			window_normalized = image_utils.normalize(window)

			batch.append({
				"window": window_normalized,
				"x": x,
				"y": y,
				"dimensions": original_dimensions
			})

			if len(batch) >= batch_size:
				windows = []
				positions = []
				dimensions = []
				for b in batch:
					windows.append(b.get("window"))
					positions.append((b.get("x"), b.get("y")))
					dimensions.append(b.get("dimensions"))
			
				windows = np.array(windows)

				predict_input_fn = tf.estimator.inputs.numpy_input_fn(
					x={"data": np.array(windows, dtype=np.float32)},
					shuffle=False
				)

				pred = estimator.predict(input_fn=predict_input_fn)

				for window, position, dimension, predict in zip(windows, positions, dimensions, pred):
					predict[predict > 0.5] = 1
					predict[predict <=  0.5] = 0

					predict = image_utils.resize(predict, (dimension[0], dimension[1]), preserve_range=True, anti_aliasing=True).astype(np.int8)			
					predict = predict.reshape((predict.shape[0], predict.shape[1]))

					predicted = image_utils.get_window(image_predicted, position[0], position[1], predict.shape[1], predict.shape[0])			
				
					if predict.shape != predicted.shape:
						import ipdb; ipdb.set_trace()
					try:
						image_utils.set_window(image_predicted, np.add(predict, predicted), position[0], position[1])
					except Exception as e:
						import ipdb; ipdb.set_trace()
				batch = []

	driver = input_dataset.GetDriver()
	output_dataset = driver.Create(output_path, image.shape[1], image.shape[0], 1, gdal.GDT_Int16)
	output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
	output_dataset.SetProjection(input_dataset.GetProjection())
	output_band = output_dataset.GetRasterBand(1)
	output_band.WriteArray(image_predicted.reshape((image_predicted.shape[0], image_predicted.shape[1])), 0, 0)
	output_band.FlushCache()

if __name__ == "__main__":
	args = arguments.parser_mode.parse_known_args(sys.argv[1:])

	if args[0].mode == "generate":
		args_generate = arguments.parser_generate.parse_args(sys.argv[1:])
		if(args_generate.image and args_generate.labels and args_generate.output and args_generate.image):

			image_utils.generate_dataset(
	                	image_path	= args_generate.image,
	                	labels_path	= args_generate.labels,
				output_path	= args_generate.output,
				chip_size	= args_generate.chip_size,
				channels	= args_generate.channels,
				grids		= args_generate.grids,
				rotate		= args_generate.rotate,
				flip		= args_generate.flip,
			)
		else:
			arguments.parser_generate.print_help()

	elif args[0].mode == "train":
		args_train = arguments.parser_train.parse_args(sys.argv[1:])
		if args_train.train and args_train.test:
			params["num_classes"] 	= args_train.classes

			train(
	                	input_train 	= args_train.train,
	                	input_test  	= args_train.test,
	                	model		= args_train.model,
	                	model_dir   	= args_train.model_dir,
	                	epochs      	= args_train.epochs,
	                	batch_size  	= args_train.batch_size,
				params          = params
	        	)
		else:
			arguments.parser_train.print_hep()
	elif args[0].mode == "evaluate":
		args_evaluate = arguments.parser_evaluate.parse_args(sys.argv[2:])
		if  args_evaluate.evaluate:
			params["num_classes"] = args_evaluate.classes

			evaluate(
	                	input_validation= args_evaluate.evaluate,
	                	model		= args_evaluate.model,
	                	model_dir 	= args_evaluate.model_dir,
				batch_size      = args_evaluate.batch_size,
				params          = params
	        	)
		else:
			arguments.parser_evaluate.print_help()
	elif args[0].mode == "predict":
		args_predict = arguments.parser_predict.parse_args(sys.argv[2:])
		if args_predict.input and args_predict.output:
			params["num_classes"]   = args_predict.classes

			predict(
	                	input_path  	= args_predict.input,
	                	output_path 	= args_predict.output,
	               		chip_size       = args_predict.chip_size,
	                        channels        = args_predict.channels,
				grids		= args_predict.grids,
	                	model		= args_predict.model,
	               		model_dir       = args_predict.model_dir,
	                        batch_size      = args_predict.batch_size,
				params          = params
	        	)
		else:
			arguments.parser_predict.print_help()
	else:
		arguments.parser.print_help()
