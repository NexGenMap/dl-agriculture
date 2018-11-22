import tensorflow as tf
import image_utils

def conv_conv_pool(input_, n_filters, mode, params, name, pool=True, activation=tf.nn.relu):
	
	net = input_

	with tf.variable_scope("layer{}".format(name)):
			for i, F in enumerate(n_filters):
					
					net = tf.layers.conv2d(
							inputs=net,
							filters=F, 
							kernel_size=3,
							activation=None,
							padding='same',
							kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5),
							kernel_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=True),
							name="conv_{}".format(i + 1))
					net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN), name="bn_{}".format(i + 1))
					net = activation(net, name="relu{}_{}".format(name, i + 1))

			if pool is False:
					return net

			pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

			return net, pool

def upconv_concat(upconv_input, concat_input, n_filter, mode, params, name, padding="same"):
	
	upconv = upconv_2D(upconv_input, n_filter, mode, params, name)

	with tf.name_scope("crop_{}".format(name)):
		concat_shape = tf.shape(concat_input)
		upconv_shape = tf.shape(upconv)

		offsets = [0, tf.cast((concat_shape[1] - upconv_shape[1]) / 2,tf.int32), tf.cast((concat_shape[2] - upconv_shape[2]) / 2,tf.int32), 0]
		size = [-1, upconv_shape[1], upconv_shape[2], n_filter]
		
		concat_input_cropped = tf.slice(concat_input, offsets, size)

	return tf.concat([upconv, concat_input_cropped], axis=-1, name="concat_{}".format(name))

def upconv_2D(tensor, n_filter, mode, params, name, activation=tf.nn.relu):
	
	net = tf.layers.conv2d_transpose(
			tensor,
			filters=n_filter,
			kernel_size=2,
			strides=2,
			activation=None,
			padding="same",
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.5),
			kernel_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=True),
			name="upsample_{}".format(name))

	net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN), name="bn_{}".format(name,))
	net = activation(net, name="relu_{}".format(name))

	return net

def twoclass_cost(y_pred, y_true):
	with tf.name_scope("cost"):
		y_pred = tf.cast(y_pred, dtype=tf.float32)
		y_true = tf.cast(y_true, dtype=tf.float32)

		logits = tf.reshape(y_pred, [-1])
		trn_labels = tf.reshape(y_true, [-1])

		intersection = tf.reduce_sum( tf.multiply(logits,trn_labels) )
		union = tf.reduce_sum( tf.subtract( tf.add(logits,trn_labels) , tf.multiply(logits,trn_labels) ) )
		loss = tf.subtract( tf.constant(1.0, dtype=tf.float32), tf.divide(intersection,union), name='loss' )

		return loss

def rescale_data_viz(data):
	return tf.divide(
							( tf.subtract(data, tf.reduce_min(data)) ),
							( tf.subtract(tf.reduce_max(data), tf.reduce_min(data)) )
					)

def model_fn(features, labels, mode, params, config):
	
	input_data = features['data']

	conv1, pool1 = conv_conv_pool(input_data, [64, 64], mode, params, name=1)
	conv2, pool2 = conv_conv_pool(pool1, [128, 128], mode, params, name=2)
	conv3, pool3 = conv_conv_pool(pool2, [256, 256], mode, params, name=3)
	conv4, pool4 = conv_conv_pool(pool3, [512, 512], mode, params, name=4)
	conv5 = conv_conv_pool(pool4, [1024, 1024], mode, params, name=5, pool=False)

	up6 = upconv_concat(conv5, conv4, 512, mode, params, name=6)
	conv6 = conv_conv_pool(up6, [512, 512], mode, params, name=6, pool=False)

	up7 = upconv_concat(conv6, conv3, 256, mode, params, name=7)
	conv7 = conv_conv_pool(up7, [256, 256], mode, params, name=7, pool=False)

	up8 = upconv_concat(conv7, conv2, 128, mode, params, name=8)
	conv8 = conv_conv_pool(up8, [128, 128], mode, params, name=8, pool=False)

	up9 = upconv_concat(conv8, conv1, 64, mode, params, name=9)
	conv9 = conv_conv_pool(up9, [64, 64], mode, params, name=9, pool=False)

	dropout = tf.layers.dropout(conv9, name="dropout", rate=params.get('dropout_rate', 0.5), training=(mode == tf.estimator.ModeKeys.TRAIN))

	output = tf.layers.conv2d(dropout, 1, (1, 1), name='output', activation=tf.nn.sigmoid, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=output)

	labels_clipped = tf.cast(image_utils.crop_tensor(labels, tf.shape(output)), dtype=tf.uint8)
	
	loss = twoclass_cost(output, labels_clipped)
	
	optimizer = tf.contrib.opt.NadamOptimizer(params.get('learn_rate', 0.00005), name='optimizer')

	with tf.name_scope("img_metrics"):

		input_data_viz = image_utils.crop_tensor(input_data[:, :, :, 0:3], tf.shape(output))
		output_viz = tf.image.grayscale_to_rgb(output[:,:,:,:])
		labels_viz = tf.multiply(tf.cast(labels_clipped[:,:,:,:], dtype=tf.uint8), 255)

		tf.summary.image('image', input_data_viz, max_outputs=params['tensorboard_images_max_outputs'])

		tf.summary.image('labels', labels_viz, max_outputs=params['tensorboard_images_max_outputs'])

		tf.summary.image('output', output_viz, max_outputs=params['tensorboard_images_max_outputs'])


	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		
	eval_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=config.model_dir+"/eval", summary_op=tf.summary.merge_all())
	return tf.estimator.EstimatorSpec(mode=mode, predictions=output, loss=loss, train_op=train_op, evaluation_hooks=[eval_summary_hook])

