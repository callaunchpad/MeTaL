import tensorflow as tf
from algorithms.architectures import feed_forward
import numpy as np
from algorithms.util import var_shape, flatten_grad, conjugate_gradient


class NGFFNetwork:
    def __init__(self, state_size, action_size, ff_hparams, lr, name='NGFFNetwork'):
        self.lr = lr

        self.s = tf.placeholder(tf.float32, [None, state_size], "state")
        self.a = tf.placeholder(tf.float32, [None, action_size], "action")
        self.r = tf.placeholder(tf.float32, [None, ], "discounted_rewards")
        self.prev_p_mu = tf.placeholder(tf.float32, [None, action_size], "previous_p_mean")
        self.prev_p_logstd = tf.placeholder(tf.float32, [None, action_size], "previous_p_std")

        with tf.variable_scope(name):
            with tf.variable_scope('network'):
                logits = feed_forward(self.s, ff_hparams)
                self.probs = tf.nn.softmax(logits)
            sigma_param = tf.get_variable('sigma', (1, action_size), tf.float32, tf.constant_initializer(0.6))

            curr_p_mu = logits
            curr_p_logstd = tf.tile(sigma_param, tf.stack((tf.shape(curr_p_mu)[0], 1)))

            prev_dist = tf.distributions.Normal(self.prev_p_mu, self.prev_p_logstd)
            curr_dist = tf.distributions.Normal(curr_p_mu, curr_p_logstd)
            fixed_curr_dist = tf.distributions.Normal(tf.stop_gradient(curr_p_mu), tf.stop_gradient(curr_p_logstd))

            prev_logp = prev_dist.log_prob(self.a)
            curr_logp = curr_dist.log_prob(self.a)

            self.var_list = tf.trainable_variables()

            surr_loss = -tf.reduce_mean(self.a * tf.exp(curr_logp - prev_logp))
            g = tf.gradients(surr_loss, self.var_list)
            self.flat_g = flatten_grad(g, self.var_list)

            fixed_kl_div = tf.contrib.distributions.kl_divergence(fixed_curr_dist, curr_dist)
            self.kl_div = tf.contrib.distributions.kl_divergence(curr_dist, prev_dist)

            grads = tf.gradients(fixed_kl_div, self.var_list)
            # tricky vector multiplication between derivative of kl divergence and gradient
            # flat_tangent is the vector we are multiplying by (the gradient)
            self.flat_tangent = tf.placeholder(tf.float32, [None])

            shapes = map(var_shape, self.var_list)
            start = 0
            tangents = []
            for shape in shapes:
                size = np.prod(shape)
                # shape the flat tangent back into shape of the gradient
                param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
                tangents.append(param)
                start += size
            # multiply gradient of KL div with reshaped tangent
            grad_prod = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
            sec_grad = tf.gradients(grad_prod, self.var_list)
            self.flat_sec_grad = flatten_grad(sec_grad, self.var_list)

    def train(self, sess, sample_s, sample_a, sample_r, sample_mu, sample_logstd):
        feed_dict = {self.s: sample_s, self.a: sample_a, self.r: sample_r,
                     self.prev_p_mu: sample_mu, self.prev_p_logstd: sample_logstd}

        # function for multiplying fisher vector with vector p
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return sess.run(self.flat_sec_grad, feed_dict)

        g = sess.run(self.flat_g, feed_dict)
        stepdir = conjugate_gradient(fisher_vector_product, -g)
        print(stepdir)


    # def train(self, sample_s, sample_a, sample_r):

    # def train(self, sample_s, sample_a, sample_r):
    #     """
    #     Trains neural network
    #     args:
    #         sample_s: sample state vectors
    #         sample_a: sample actions (integers)
    #         sample_r: sample rewards (floats)
    #     Returns:
    #         Error value for the sample batch
    #     """
    #     feed_dict = {self.s: sample_s, self.a: sample_a, self.r: sample_r}
    #     error, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
    #     return error
    #
    #
    # def action_dist(self, state):
    #     """
    #     Outputs action distribution based on state
    #     args:
    #         state: current state vector
    #     Returns:
    #         Vector of action distributions
    #     """
    #     return self.sess.run(self.outputs, feed_dict={self.s: state})

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
    sam_logstd = [[0.04, 0.02], [0.03, 0.01]]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        agent.train(sess, sam_s, sam_a, sam_r, sam_mu, sam_logstd)
