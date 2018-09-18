"""
Functions that build the graph for specific architectures

TODO: Build tests

@Authors: Arsh Zahed, Yi Liu
"""

import tensorflow as tf


def FeedForward(_input, hparams, name="ffn"):
	"""
	Builds a Feed Forward NN with linear output

	Args:
		_input: Tensor of shape [None, input_size]
		hparams: Dictionary of hyperparameters
			'output_size': Dimensionality of output
			'hidden_sizes': List of hidden layer sizes
			'activations': List of activation functions for each layer
	Returns:
		Output tensor of shape [None, output_size]
	@Authors: Arsh Zahed
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


def MakeRNNCell(rnn_layer_sizes,
                dropout_keep_prob=1.0,
                attn_length=0,
                base_cell=tf.contrib.rnn.BasicLSTMCell,
                residual_connections=False,
                activation=tf.nn.tanh):
	"""
	Makes an RNN cell from the given hyperparameters. (From Magenta)

	Args:
		rnn_layer_sizes: A list of integer sizes (in units) for each layer of
		    the RNN.
		dropout_keep_prob: The float probability to keep the output of any
		    given sub-cell.
		attn_length: The size of the attention vector.
		base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.

	Returns:
		A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
	@Authors: Arsh Zahed
	"""
	cells = []
	for i in range(len(rnn_layer_sizes)):
		cell = base_cell(rnn_layer_sizes[i], activation=activation)
		if attn_length and not cells:
		  # Add attention wrapper to first layer.
		  cell = tf.contrib.rnn.AttentionCellWrapper(
		      cell, attn_length, state_is_tuple=True)
		if residual_connections:
		  cell = tf.contrib.rnn.ResidualWrapper(cell)
		  if i == 0 or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
		    cell = tf.contrib.rnn.InputProjectionWrapper(cell, rnn_layer_sizes[i])
		cell = tf.contrib.rnn.DropoutWrapper(
		    cell, output_keep_prob=dropout_keep_prob)
		cells.append(cell)

	cell = tf.contrib.rnn.MultiRNNCell(cells)

	return cell


def DynamicRNN(_input, hparams, initial_state=None, name="lstm"):
	"""
	Builds andand executes Dynamic RNN with specified activation

	Args:
		_input: Tensor of shape [None, total_time, input_size]
		hparams: Dictionary of hyperparameters
			'rnn_layer_sizes': List of RNN layer sizes
			'dropout_keep_prob': Probability of not dropping
			'attn_length': Integer length of attention
			'base_cell': RNN Cell class from tf.contrib.rnn.*
			'residual_connections': Boolean, True to have residuals
			'activation': Output activation of RNN
	Returns:
		Outputs and states Tensors. Output Tensor of shape 
			[None, total_time, rnn_layer_sizes[-1]]
			State Tensor (tuples) match shapes specified in hyperparameters
	@Authors: Arsh Zahed
	"""
	
	# Set defaults if they dont exist in hparams
	if 'dropout_keep_prob' not in hparams:
		hparams['dropout_keep_prob'] = 1.0
	if 'attn_length' not in hparams:
		hparams['attn_length'] = 0
	if 'base_cell' not in hparams:
		hparams['base_cell'] = tf.contrib.rnn.BasicLSTMCell
	if 'residual_connections' not in hparams:
		hparams['residual_connections'] = False
	if 'activation' not in hparams:
		hparams['activation'] = tf.tanh

	# Build RNN Cell
	with tf.variable_scope(name):
		rnn_cell = MakeRNNCell(hparams['rnn_layer_sizes'],
			                   hparams['dropout_keep_prob'],
			                   hparams['attn_length'],
			                   hparams['base_cell'],
			                   hparams['residual_connections'],
			                   hparams['activation'])

	outputs, states = tf.nn.dynamic_rnn(rnn_cell, _input, initial_state=initial_state,
										dtype=_input.dtype)

	return outputs, states


def CNN(_input, hparams, name="cnn"):
	"""
	Builds a Convolutional Neural Network with a flattened output

	Args:
		_input: Tensor of shape [None, image_height, image_width, channels]
		hparams: Dictionary of hyperparameters
			'feature_maps': List of feature maps for each layer
			'kernel_sizes': List of kernel sizes for each layer
			'stride_lengths': List of strides for each layer
			'padding_types': List of padding for each layer
			'activations': List of activation functions for each layer
	Returns:
		Flattened output tensor of shape [None, output_size]
	@Authors: Yi Liu
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


def RCNN(self, _input, hparams, name='rcnn'):
	# TODO
	raise NotImplementedError('RCNN not implemented')
