import numpy as np
import scipy as scipy
import tensorflow as tf

def conj_grad_func(A, b):
	"""
	function that calls out to scipy function for solving Ax=b with conjugate gradient method
	@Author Hank O'Brien
	"""
	#make sure that A is n by n and b is n by 1
	assert(np.shape(A)[0] == np.shape(A)[1] and np.shape(b)[0] == np.shape(A)[0] and np.shape(b)[1] == 1)
	return np.array([scipy.sparse.linalg.cg(A,b)[0]]).T

# there may be some issues with distributed training when using this function
A = tf.placeholder(dtype=tf.float32, shape=[None, None])
b = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# conj_grad_op = tf.py_func(conj_grad_func, [A, b], tf.float32, stateful=False, name='conj_grad_func')

def conj_grad_wrapper(A,b):
	return tf.py_func(conj_grad_func, [A, b], tf.float32, stateful=False, name='conj_grad_func')

if __name__ == '__main__':

	# A_placeholder = tf.placeholder(tf.float32, shape=[3,3])
	# b_placeholder = tf.placeholder(tf.float32, shape=[3,1])

	output = conj_grad_wrapper(A, b)

	# with tf.Graph().as_default():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		A_input = np.array([[1,0,2],[3,4,5],[6,9,1]])
		A_input = A_input.T @ A_input
		b_input = np.array([[1,2,3]]).T

		x_out = sess.run([output], feed_dict={A: A_input, b: b_input})

		
		if(np.allclose(x_out[0], np.linalg.solve(A_input,b_input), rtol=0.0001, atol=0.0001, equal_nan=False)):
			print("passed test")
		else:
			raise AssertionError('output was not expected')
