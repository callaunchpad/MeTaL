"""
Functions that build the graph for specific architectures

"""

import tensorflow as tf


def FeedForward(self, _input, hparams, name="ffn"):
		"""
		Builds a Feed Forward NN with linear output

		Args:
			_input: Tensor of shape [None, input_size]
			hparams: Dictionary of hyperparameters
				'output_size': Dimensionality of output
				'hidden_sizes': List of hidden layer sizes
				'activations': List of activation functions for each layer
		@Author: Arsh Zahed
		"""

		# We iteratively nest the layers
		net = _input
		hidden_sizes = hparams['hidden_sizes']
		activations = hparams['activations']
		with tf.variable_scope(name):
			for i in range(len(activations)):
				net = tf.layers.dense(net, hidden_sizes[i], activations[i])
			# Call our prediction/policy y_hat. 
			# Linear activation allows for logits
			y_hat = tf.layers.dense(net, hparams['output_size'])

		return y_hat


def LSTM(self, _input, hparams, name="lstm"):
	# TODO
	raise NotImplementedError('AHH')


def CNN(self, _input, hparams, name="cnn"):
	"""
	Builds a Convolutional NN with a flattened output

	Args:
		_input: Tensor of shape [None, image_height, image_width, channels]
		hparams: Dictionary of hyperparameters
			'feature_maps': List of feature maps for each layer
			'kernel_sizes': List of kernel sizes for each layer
			'stride_lengths': List of strides for each layer
			'padding_types': List of padding for each layer
			'activations': List of activation functions for each layer
	@Author: Yi Liu
	"""

	net = _input
	feature_maps = hparams['feature_maps']
	kernel_sizes = hparams['kernel_sizes']
	stride_lengths = hparams['stride_lengths']
	padding_types = hparams['padding_types']
	activations = hparams['activations']

	with tf.variable_scope(name):
		for i in range(len(activations)):
			net = tf.layers.conv2d(
				inputs=net,
				filters=feature_maps[i],
				kernel_size=kernel_sizes[i],
				strides=stride_lengths[i],
				padding=padding_types[i],
				activation=activations[i]
			)
		# Flatten network
		flat = tf.contrib.layers.flatten(net)
	return flat
