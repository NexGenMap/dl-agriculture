import tensorflow as tf

_L2_WEIGHT_REGULARIZER=0.01

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
				kernel_regularizer=tf.contrib.layers.l2_regularizer(_L2_WEIGHT_REGULARIZER),
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
		kernel_regularizer=tf.contrib.layers.l2_regularizer(_L2_WEIGHT_REGULARIZER),
		name="upsample_{}".format(name))

def model_fn(features, labels, mode, params, config):
	"""
	https://github.com/zhixuhao/unet/blob/master/model.py
	"""
	input_data = features['data']
	flags = params

	conv1 = conv_conv(input_data, [32, 32], mode, flags, name=1)
	pool1 = pool(conv1, mode, name=1)

	conv2 = conv_conv(pool1, [64, 64], mode, flags, name=2)
	pool2 = pool(conv2, mode, name=2)

	conv3 = conv_conv(pool2, [128, 128], mode, flags, name=3)
	pool3 = pool(conv3, mode, name=3)

	conv4 = conv_conv(pool3, [256, 256], mode, flags, name=4)
	drop4 = dropout(conv4, 0.5, mode, name=4)
	pool4 = pool(drop4, mode, name=4)

	conv5 = conv_conv(pool4, [512, 512], mode, flags, name=5)
	drop5 = dropout(conv5, 0.5, mode, name=5)

	up6   = upconv_concat(drop5, drop4, 256, flags, name=6)
	conv6 = conv_conv(up6, [256, 256], mode, flags, name=6)
	
	up7   = upconv_concat(conv6, conv3, 128, flags, name=7)
	conv7 = conv_conv(up7, [128, 128], mode, flags, name=7)

	up8   = upconv_concat(conv7, conv2, 64, flags, name=8)
	conv8 = conv_conv(up8, [64, 64], mode, flags, name=8)
	
	up9   = upconv_concat(conv8, conv1, 32, flags, name=9)
	conv9 = conv_conv(up9, [32, 32], mode, flags, name=9)

	logits = tf.layers.conv2d(conv9, params['num_classes'], (1, 1), name='logits', activation=None, padding='same')
		
	pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
	
	pred_decoded_labels = tf.cast(pred_classes, tf.uint8)	

	predictions = {
		'classes': pred_classes,
		'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
		'decoded_labels': pred_decoded_labels
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes)
	
	labels = tf.squeeze(labels, axis=3)
	
	logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
	
	labels_flat = tf.reshape(labels, [-1, ])


	valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
	valid_logits  = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
	valid_labels  = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
	
	preds_flat = tf.reshape(pred_classes, [-1, ])
	valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
	confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

	predictions['valid_preds'] = valid_preds
	predictions['valid_labels'] = valid_labels
	predictions['confusion_matrix'] = confusion_matrix


	# Create a tensor named train_accuracy for logging purposes
	accuracy = tf.metrics.accuracy(valid_labels, valid_preds)
	mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
	metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}
	tf.identity(accuracy[1], name='train_px_accuracy')
	tf.summary.scalar('train_px_accuracy', accuracy[1])
	
	regularization_loss = tf.losses.get_regularization_losses()
	cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits_by_num_classes, labels=tf.cast(labels_flat,tf.int32))
	loss = cross_entropy+sum(regularization_loss)
	optimizer = tf.contrib.opt.NadamOptimizer(params.get('learning_rate', 0.0001), name='optimizer')
	
	input_data_viz = ((input_data[:, :, :, 0:3]))
	output_viz = tf.cast(pred_classes * 255, dtype=tf.uint8)
	labels_viz = tf.cast(tf.expand_dims(labels, -1), dtype=tf.uint8) * 255

	tf.summary.scalar('cross_entropy', cross_entropy)

	tf.summary.image('image', input_data_viz, max_outputs=params['tensorboard_images_max_outputs'])
	tf.summary.image('labels', labels_viz, max_outputs=params['tensorboard_images_max_outputs'])
	tf.summary.image('output', output_viz, max_outputs=params['tensorboard_images_max_outputs'])

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

	eval_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=config.model_dir + "/eval",  summary_op=tf.summary.merge_all())
	
	return tf.estimator.EstimatorSpec(
		mode=mode, 
		predictions=pred_classes, 
		loss=loss, 
		train_op=train_op, 
		eval_metric_ops=metrics,
		evaluation_hooks=[eval_summary_hook]
	)


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
