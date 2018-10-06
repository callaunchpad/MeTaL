import numpy as np
import scipy as scipy
from scipy.sparse.linalg import cg
import tensorflow as tf

def conj_grad_func(A, b):
	"""
	function that calls out to scipy function for solving Ax=b with conjugate gradient method
	@Author Hank O'Brien
	"""
	#make sure that A is n by n and b is n by 1
	assert(np.shape(A)[0] == np.shape(A)[1] and np.shape(b)[0] == np.shape(A)[0] and np.shape(b)[1] == 1)
	return np.array([cg(A,b)[0]]).T

def conj_grad_wrapper(A,b):
	"""
	wrapper for tf.py_func so that we can pass dynamic A and b values
	@Author Hank O'Brien
	"""
	# there may be some issues with distributed training when using this function
	return tf.py_func(conj_grad_func, [A, b], tf.float32, stateful=False, name='conj_grad_func')

A = tf.placeholder(dtype=tf.float32, shape=[None, None])
b = tf.placeholder(dtype=tf.float32, shape=[None, 1])

if __name__ == '__main__':
	output = conj_grad_wrapper(A, b)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#an arbitrary matrix
		A_input = np.array([[1,0,2],[3,4,5],[6,9,1]])
		#make the matrix positive definite
		A_input = A_input.T @ A_input
		#arbitrary output
		b_input = np.array([[1,2,3]]).T
		#computed output
		x_out = sess.run([output], feed_dict={A: A_input, b: b_input})

		#test if computed output is the same as 'correct' output (found through other technique)
		if(np.allclose(x_out[0], np.linalg.solve(A_input,b_input), rtol=0.0001, atol=0.0001, equal_nan=False)):
			print("passed test")
		else:
			raise AssertionError('output was different than expected')
