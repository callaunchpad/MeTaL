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

		# Placeholder for input

		# We iteratively nest the layers
		net = _input
		hidden_sizes = hparams['hidden_sizes']
		activations = hparams['activations']
		with tf.variable_scope('dense_layers'):
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
	# TODO
	raise NotImplementedError('AHH')