import tensorflow as tf

def conv_conv(input_, n_filters, mode, flags, name, activation=tf.nn.relu):
	net = input_

	with tf.variable_scope("layer{}".format(name)):
		for i, F in enumerate(n_filters):
			net = tf.layers.conv2d(
				inputs=net,
				filters=F,
				kernel_size=(3, 3),
				activation=None,
				padding='same',
				kernel_regularizer=tf.contrib.layers.l2_regularizer(0.30),
				kernel_initializer=tf.initializers.variance_scaling(scale=0.01, distribution="normal"),
				name="conv_{}".format(i + 1))
			net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN), name="bn_{}".format(i + 1))
			net = activation(net, name="relu{}_{}".format(name, i + 1))

		return net

def pool(input_, mode, name):
	return tf.layers.max_pooling2d(input_, (2, 2), strides=(2, 2), name="pool_{}".format(name))

def dropout(input_, rate, mode, name):
	return tf.layers.dropout(inputs=input_, rate=rate, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout_{}".format(name))

def upconv_concat(inputA, input_B, n_filter, flags, name):
	up_conv = upconv_2D(inputA, n_filter, flags, name)
	return tf.concat([up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name):
	return tf.layers.conv2d_transpose(
		tensor,
		filters=n_filter,
		kernel_size=2,
		strides=2,
		kernel_regularizer=tf.contrib.layers.l2_regularizer(0.30),
		kernel_initializer=tf.initializers.variance_scaling(scale=0.01, distribution="normal"),
		name="upsample_{}".format(name))


def twoclass_cost(y_pred, y_true):
	with tf.name_scope("cost"):
		y_pred = tf.cast(y_pred, dtype=tf.float32)
		y_true = tf.cast(y_true, dtype=tf.float32)

		logits = tf.reshape(y_pred, [-1])
		trn_labels = tf.reshape(y_true, [-1])

		intersection = tf.reduce_sum(tf.multiply(logits, trn_labels))
		union = tf.reduce_sum(tf.subtract(tf.add(logits, trn_labels), tf.multiply(logits, trn_labels)))
		loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.divide(intersection, union))

		return loss


def multiclass_cost(y_pred, y_true):
	with tf.name_scope("cost"):
		loss = tf.losses.mean_squared_error(y_true, y_pred)
		return loss


def description(features, labels, mode, params, config):
	"""
	https://github.com/zhixuhao/unet/blob/master/model.py
	"""
	input_data = features['data']
	flags = params

	conv1 = conv_conv(input_data, [64, 64], mode, flags, name=1)
	pool1 = pool(conv1, mode, name=1)

	conv2 = conv_conv(pool1, [128, 128], mode, flags, name=2)
	pool2 = pool(conv2, mode, name=2)

	conv3 = conv_conv(pool2, [256, 256], mode, flags, name=3)
	pool3 = pool(conv3, mode, name=3)

	conv4 = conv_conv(pool3, [512, 512], mode, flags, name=4)
	drop4 = dropout(conv4, 0.5, mode, name=4)
	pool4 = pool(drop4, mode, name=4)

	conv5 = conv_conv(pool4, [1024, 1024], mode, flags, name=5)
	drop5 = dropout(conv5, 0.5, mode, name=5)

	up6   = upconv_concat(drop5, drop4, 512, flags, name=6)
	conv6 = conv_conv(up6, [512, 512], mode, flags, name=6)
	
	up7   = upconv_concat(conv6, conv3, 256, flags, name=7)
	conv7 = conv_conv(up7, [256, 256], mode, flags, name=7)

	up8   = upconv_concat(conv7, conv2, 128, flags, name=8)
	conv8 = conv_conv(up8, [128, 128], mode, flags, name=8)
	
	up9   = upconv_concat(conv8, conv1, 64, flags, name=9)
	conv9 = conv_conv(up9, [64, 64], mode, flags, name=9)

	output = tf.layers.conv2d(conv9, 1, (1, 1), name='output', activation=tf.nn.sigmoid, padding='same', kernel_initializer=tf.initializers.variance_scaling(scale=0.01, distribution="normal"))

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=output)
	
	loss =  twoclass_cost(output, labels)
	optimizer = tf.contrib.opt.NadamOptimizer(0.00005, name='optimizer')

	with tf.name_scope("img_metrics"):
		input_data_viz = ((input_data[:, :, :, 0:3]))
		output_viz = tf.image.grayscale_to_rgb(output[:,:,:,:])
		labels_viz = tf.multiply(tf.cast(labels[:,:,:,:], dtype=tf.uint8), 255)

		#conv1_viz = tf.image.grayscale_to_rgb(put_features_on_grid(conv1)[:, :, :, :])

		tf.summary.image('img', input_data_viz, max_outputs=1)
		tf.summary.image('output', output_viz, max_outputs=1)
		tf.summary.image('labels', labels_viz, max_outputs=1)
		#tf.summary.image('conv2', conv1_viz, max_outputs=1)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

	eval_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=config.model_dir + "/eval",  summary_op=tf.summary.merge_all())
	return tf.estimator.EstimatorSpec(mode=mode, predictions=output, loss=loss, train_op=train_op, evaluation_hooks=[eval_summary_hook])


def put_features_on_grid(features):
	iy=tf.shape(features, out_type=tf.int32)[1]
	ix=tf.shape(features, out_type=tf.int32)[2]
	n_ch=tf.cast(tf.shape(features, out_type=tf.int32)[3], tf.float32)
	b_size=tf.shape(features, out_type=tf.int32)[0]
	square_size=tf.cast(tf.ceil(tf.sqrt(n_ch)),tf.float32)
	z_pad=tf.cast(tf.round(square_size**2-n_ch), tf.int32)
	black=tf.minimum(0.0,tf.reduce_min(features))
	pad=1+tf.cast((ix/64), tf.int32)
	square_size=tf.cast(square_size, tf.int32)
	features = tf.pad(features, [[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=black)
	features = tf.reshape(features,[b_size,iy,ix,square_size,square_size])
	features = tf.pad(features, [[0,0],[pad,0],[pad,0],[0,0],[0,0]], mode='constant',constant_values=black)
	iy+=pad
	ix+=pad
	features = tf.transpose(features,(0,3,1,4,2))
	features = tf.reshape(features,[-1,square_size*iy,square_size*ix,1])
	return tf.pad(features, [[0,0],[0,pad],[0,pad],[0,0]], mode='constant',constant_values=black)
