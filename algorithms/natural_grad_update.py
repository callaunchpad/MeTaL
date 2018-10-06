import numpy as np
import scipy
import tensorflow as tf

def conj_grad_func(A, b):
	"""
	function that calls out to scipy function for solving Ax=b with conjugate gradient method
	@Author Hank O'Brien
	"""
	#make sure that A is n by n and b is n by 1
	assert(np.shape(A)[0] == np.shape(A)[1] and np.shape(b)[0] == np.shape(A)[0] and np.shape(b)[1] == 1)
	return scipy.sparse.linalg.cg(A,b)

# there may be some issues with distributed training when using this function
input_placeholder = tf.placeholder(dtype=tf.float32, [None, None])

conj_grad_op = tf.py_func(conj_grad_func, [input_placeholder], tf.float32)