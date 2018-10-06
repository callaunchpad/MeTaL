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
        # previous policy's mean
        self.prev_p_mu = tf.placeholder(tf.float32, [None, action_size], "previous_p_mean")
        # previous policy's log standard deviation
        self.prev_p_logstd = tf.placeholder(tf.float32, [None, action_size], "previous_p_std")

        with tf.variable_scope(name):
            # create network
            with tf.variable_scope('network'):
                logits = feed_forward(self.s, ff_hparams)
                self.probs = tf.nn.softmax(logits)
            # create a single layer of logstd
            sigma_param = tf.get_variable('sigma', (1, action_size), tf.float32, tf.constant_initializer(0.6))

            # current policy's means (logits)
            curr_p_mu = logits
            # combine layers of logstd into batch size
            curr_p_logstd = tf.tile(sigma_param, tf.stack((tf.shape(curr_p_mu)[0], 1)))

            # create normal distributions for previous policy
            prev_dist = tf.distributions.Normal(self.prev_p_mu, self.prev_p_logstd)
            # create normal distributions for current policy
            curr_dist = tf.distributions.Normal(curr_p_mu, curr_p_logstd)
            fixed_curr_dist = tf.distributions.Normal(tf.stop_gradient(curr_p_mu), tf.stop_gradient(curr_p_logstd))

            # log probabilities of actions for each policy distribution
            prev_logp = prev_dist.log_prob(self.a)
            curr_logp = curr_dist.log_prob(self.a)

            # get list of trainable variables in the network
            self.var_list = tf.trainable_variables()

            # surrogate loss is the difference between policies multiplied by advantage
            surr_loss = -tf.reduce_mean(self.r * tf.exp(curr_logp - prev_logp))
            # create g used in inverse(fisher) * g
            g = tf.gradients(surr_loss, self.var_list)
            # flatten gradient
            self.flat_g = flatten_grad(g, self.var_list)

            # calculate kl divergence of policy with itself
            fixed_kl_div = tf.contrib.distributions.kl_divergence(fixed_curr_dist, curr_dist)
            # kl divergence between current and previous policies (only used for logging)
            self.kl_div = tf.contrib.distributions.kl_divergence(curr_dist, prev_dist)

            # first derivative of kl divergence
            grads = tf.gradients(fixed_kl_div, self.var_list)
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

            self.true_lr = tf.placeholder(tf.float32, shape=())
            self.true_grad = tf.placeholder(tf.float32, shape=(-1))

            opt = tf.train.GradientDescentOptimizer(self.true_lr)
            self.train_op = opt.apply_gradients([(true_grad_list, self.var_list)])

    def train(self, sess, sample_s, sample_a, sample_r, sample_mu, sample_logstd):
        """
            Trains neural network
            args:
                sess: tensorflow session
                sample_s: sample state vectors
                sample_a: sample actions (integers)
                sample_r: sample rewards (floats)
                sample_mu: means of policy distribution
                sample_logstd: log standard deviation of distribution
        """
        feed_dict = {self.s: sample_s, self.a: sample_a, self.r: sample_r,
                     self.prev_p_mu: sample_mu, self.prev_p_logstd: sample_logstd}

        # function for multiplying fisher vector with vector p
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return sess.run(self.flat_sec_grad, feed_dict)

        # get g vector
        g = sess.run(self.flat_g, feed_dict)
        # calculate step size
        stepdir = conjugate_gradient(fisher_vector_product, -g)
        step_size = np.sqrt(self.lr*2/np.dot(g, stepdir))
        
        feed_dict = {self.true_lr: step_size, self.true_grad: stepdir}
        sess.run(self.train_op(), feed_dict)

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
