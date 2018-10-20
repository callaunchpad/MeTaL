"""
Creates natural gradient network class

@Authors: Yi Liu, Andrew Dickson
"""

import tensorflow as tf
from algorithms.architectures import feed_forward
import numpy as np
from algorithms.util import var_shape, flatten_grad, conjugate_gradient


class NGFFNetwork:
    def __init__(self, state_size, action_size, ff_hparams, lr, name='NGFFNetwork'):
        self.lr = lr

        self.s = tf.placeholder(tf.float32, [None, state_size], "state")
        self.a = tf.placeholder(tf.int32, [None, ], "action")
        self.r = tf.placeholder(tf.float32, [None, ], "discounted_rewards")
        # previous policy's distribution
        self.prev_p_dist = tf.placeholder(tf.float32, [None, action_size], "previous_p_dist")

        with tf.variable_scope(name):
            # create network
            with tf.variable_scope('network'):
                self.logits = feed_forward(self.s, ff_hparams)
                self.probs = tf.nn.softmax(self.logits)

            one_hot = tf.one_hot(self.a, action_size)

            # current policy distribution
            curr_p_dist = tf.stop_gradient(self.probs)

            # log probabilities of actions for each policy distribution
            #prev_logp = prev_dist.log_prob(self.a)
            #curr_logp = curr_dist.log_prob(self.a)
            #print("prev_logp", prev_logp)
            #print("curr_logp", curr_logp)

            # get list of trainable variables in the network
            self.var_list = tf.trainable_variables()

            # surrogate loss is the difference between policies multiplied by advantage
            #surr_loss = -tf.reduce_mean(self.r * tf.exp(curr_logp - prev_logp))
            # create g used in inverse(fisher) * g

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.logits)
            loss = tf.reduce_mean(cross_entropy * self.r)

            g = tf.gradients(loss, self.var_list)

            # flatten gradient
            self.flat_g = flatten_grad(g, self.var_list)

            # calculate kl divergence of policy with itself. 
            kl_plus_const = tf.nn.softmax_cross_entropy_with_logits_v2(labels=curr_p_dist, logits=self.logits)

            # first derivative of kl divergence
            grads = tf.gradients(kl_plus_const, self.var_list)
            # tricky vector multiplication between derivative of kl divergence and gradient
            # flat_tangent is the vector we are multiplying by (the gradient)
            self.flat_tangent = tf.placeholder(tf.float32, [None])

            shapes = map(var_shape, self.var_list)
            start = 0
            tangents = []
            # for loop shapes flat tangent into shapes of network
            for shape in shapes:
                size = np.prod(shape)
                # shape the flat tangent back into shape of the gradient
                param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
                tangents.append(param)
                start += size
            # multiply gradient of KL div with reshaped tangent
            grad_prod = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
            # second gradient (fisher)
            sec_grad = tf.gradients(grad_prod, self.var_list)
            self.flat_sec_grad = flatten_grad(sec_grad, self.var_list)

            # parameter manipulation
            self.p_params = tf.concat([tf.reshape(v, [np.prod(var_shape(v))]) for v in self.var_list], 0)
            # assignment operator
            shapes = list(map(var_shape, self.var_list))  # note, here is the needed change.
            total_size = sum(np.prod(shape) for shape in shapes)
            self.new_params = tf.placeholder(tf.float32, [total_size])
            start = 0
            assigns = []
            for (shape, v) in zip(shapes, self.var_list):
                size = np.prod(shape)
                assigns.append(tf.assign(v, tf.reshape(self.new_params[start:start + size], shape)))
                start += size
            self.assign_ops = tf.group(*assigns)

            # self.true_lr = tf.placeholder(tf.float32, shape=())
            # self.true_grad = tf.placeholder(tf.float32, shape=(-1))
            #
            # opt = tf.train.GradientDescentOptimizer(self.true_lr)
            # self.train_op = opt.apply_gradients([(true_grad_list, var_list)])

    def action_dist(self, state, sess):
        """
        Outputs action distribution based on state
        args:
            state: current state vector
            sess: tf.Session to run in
        Returns:
            Vector of action distributions
        """
        return sess.run(self.probs, feed_dict={self.s: state})

    def train(self, sample_s, sample_a, sample_r, sess):
        """
            Trains neural network
            args:
                sess: tensorflow session
                sample_s: sample state vectors
                sample_a: sample actions (integers)
                sample_r: sample rewards (floats)
        """
        feed_dict = {self.s: sample_s, self.a: sample_a, self.r: sample_r}

        # function for multiplying fisher vector with vector p
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return sess.run(self.flat_sec_grad, feed_dict)

        # get g vector
        g = sess.run(self.flat_g, feed_dict)
        # calculate step size
        stepdir = conjugate_gradient(fisher_vector_product, g)
        step_size = np.sqrt(2 * self.lr / stepdir.dot(g))

        # assign new parameters
        params = sess.run(self.p_params, feed_dict)
        new_params = params + step_size * stepdir
        sess.run(self.assign_ops, feed_dict={self.new_params: new_params})
        return np.sum(step_size)


if __name__ == "__main__":
    ff_hparams = {
        'hidden_sizes': [30],
        'activations': [tf.nn.leaky_relu],
        'output_size': 2,
        'kernel_initializers': [tf.contrib.layers.xavier_initializer(),
                                tf.contrib.layers.xavier_initializer()]
    }
    agent = NGFFNetwork(4, 2, ff_hparams, 0.01)
    sam_s = [[0.4, 0.3, 0.6, 0.1], [0.1, 0.3, 0.6, 0.1]]
    sam_a = [[0.3, 0.2], [0.1, 0.6]]
    sam_r = [0.6, 0.3]
    sam_mu = [[0.3, 0.2], [0.2, 0.5]]
    sam_logstd = [[0.4, 0.2], [0.3, 0.1]]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        agent.train(sam_s, sam_a, sam_r, sess)
