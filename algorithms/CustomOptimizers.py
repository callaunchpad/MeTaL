from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np
from natural_grad_update import conj_grad_wrapper


class NaturalGradientOptimizer(optimizer.Optimizer):
    """Implementation of PowerSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """
    def __init__(self, outputs, learning_rate=0.001, use_locking=False, name="NaturalGradientOptimizer"):
        super(NaturalGradientOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._outputs_t = outputs #It might be bad to pass these in. TODO fix hack. 
        
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

    	fisher_matrix = #TODO

        #The vector which produces the gradient when multiplied by the fisher matrix. 
        inverse_gradient = conj_grad_wrapper(fisher_matrix, grad)


        inv_multiplier = tf.stop_gradient(tf.sqrt(tf.matmul(inverse_gradient, grad, transpose_a = True) / (2*lr_t)))

        var_update = state_ops.assign_sub(var, inverse_gradient/inv_multiplier)
        #Create an op that groups multiple operations.
        #When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update])

     def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")