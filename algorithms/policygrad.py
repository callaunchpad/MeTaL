"""
Classes for policy gradient neural networks
@Authors: Yi Liu
"""
import tensorflow as tf
from algorithms.architectures import feed_forward


class PGFFNetwork:
    """
    Creates a policy gradient feed forward neural network
    args:
            n_episodes:   How many episodes do we run each game, aka batch_size
            bin_size:     Vector of length state_size, detailing bin sizes for each dimension for converting continuous states to discrete.
    Returns:
            init function, void return
    @Authors: Yi Liu, Jihan Yin, Joey Hejna
    """
    def __init__(self, sess, state_size, action_size, ff_hparams, lr, n_episodes, bin_size, name='PGFFNetwork'):
        """
        Creates network
        args:
            sess: session
            state_size: number of values in the state vector
            action_size: number of possible discrete actions
            ff_hparams: hyper parameters for the network
            lr: learning rate of network
            name: variable scope
        """
        # learning rate
        self.lr = lr
        self.sess = sess
        self.n_episodes = n_episodes
        
        # Expected total rewards from state s. We update this as we train
        self.expected_rewards = {}
        self.bin_size = bin_size

        # list of states for a single game
        self.state_size = state_size
        self.s = tf.placeholder(tf.float32, [self.n_episodes, None, state_size], "state")
        # list of actions (integers) for a single game
        self.a = tf.placeholder(tf.int32, [self.n_episodes, None, action_size], "action")
        self.action_size = action_size
        
        # list of discounted rewards (floats) for a single game
        self.r = tf.placeholder(tf.float32, [self.n_episodes, None,], "discounted_rewards")

        with tf.variable_scope(name):
            with tf.variable_scope('network'):
                # logits - output of the network without
                logits = feed_forward(self.s, ff_hparams)
                # softmax layer to create probability array
                self.outputs = tf.nn.softmax(logits)

            with tf.variable_scope('training'):
                # onehot encoding of the actions
                one_hot = tf.one_hot(self.a, action_size)
                # determine cross entropy
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=logits)
                
                self.loss = tf.reduce_mean(cross_entropy * self.r)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sample_s, sample_a, sample_r, use_baseline=False):
        """
        Trains neural network with batch of episodes
        args:
            sample_s:       self.n_episodes sequences of sample state vectors
            sample_a:       self.n_episodes sequences of sample actions (integers)
            sample_r:       self.n_episodes sequences of sample rewards (floats)
            use_baseline:   Flag on whether we should subtract a baseline or not for variance reduction
        Returns:
            Error value for the sample batch
        @Authors: Jihan Yin
        """
        # Update expected rewards, using exponentional moving average. We use this as the baseline
        if use_baseline:
            tmp_expected_rewards = {}
            num_samples_state = {}  # how many times we encounter each state, used for averaging
            for states, actions, rewards in zip(sample_s, sample_a, sample_r):  # go through the batch
                for state, action, reward in zip(states, actions, rewards): # go through the sample
                    disc_state = self.discretize(state)
                    if disc_state not in tmp_expected_rewards:
                        tmp_expected_rewards[disc_state] = 0
                    tmp_expected_rewards[disc_state] += reward
                    num_samples_state[disc_state] += 1

            for s in tmp_expected_rewards: # get averages
                tmp_expected_rewards[s] /= num_samples_state[s]
                if s not in self.expected_rewards:
                    self.expected_rewards[s] = 0
                self.expected_rewards[s] = 0.5 * (self.expected_rewards[s] + tmp_expected_rewards[s]) # exp moving average
        
            for batch_i in range(sample_s): # go through the batch
                for step_i in range(sample_s[batch_i]): # go through the sample
                    sample_r[batch_i][step_i] -= self.expected_rewards[self.discretize(sample_s[batch_i][step_i])] # subtract baseline
            
        # Pad all samples to have same length as max. Add zeros for padding.
        maxlen = max([len(s) for s in sample_s])
        for i in range(self.n_episodes):
            sample_s[i] = sample_s[i] + np.zeros([maxlen-len(sample_s[i]),self.state_size]).tolist()
            sample_a[i] = sample_a[i] + np.zeros([maxlen-len(sample_a[i]),self.action_size]).tolist()
            sample_r[i] = sample_r[i] + np.zeros(maxlen-len(sample_r[i])).tolist()

        
        feed_dict = {self.s: sample_s, self.a: sample_a, self.r: sample_r}
        error, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return error


    def action_dist(self, state, sess):
        """
        Outputs action distribution based on state
        args:
            state: current state vector
            sess: tf.Session to run in
        Returns:
            Vector of action distributions
        """
        return sess.run(self.outputs, feed_dict={self.s: state})


    def discretize(self, state):
        """
        Outputs discrete version of continuous state
        args:
            state: current state vector in its continuous form
        Returns:
            state vector in discrete forms
        Example:
            If the bin_size for state i is 0.1, and state i = 2.16, we round it to i = 2.2
        @Authors: Jihan Yin
        """
        discretized_state = []
        for b, s in zip(bin_size, state):
            discretized_state.append(round(s/b)*b)
        return discretized_state



class PGLSTM:
    """
    TODO
    """

    def __init__(self):
        return


class PGConvNetwork:
    """
    A basic network that performs convolutions. (Temporary!!)
    """

    def __init__(self, state_size, action_size, learning_rate, name='PGConvNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ],
                                                                  name="discounted_episode_rewards_")

                # Add this placeholder for having this variable in tensorboard
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            with tf.name_scope("conv1"):
                """
                First convnet:
                CNN
                BatchNormalization
                ELU
                """
                # Input is 84x84x4
                self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                              filters=32,
                                              kernel_size=[8, 8],
                                              strides=[4, 4],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv1")

                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name='batch_norm1')

                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
                ## --> [20, 20, 32]

            with tf.name_scope("conv2"):
                """
                Second convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                              filters=64,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv2")

                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name='batch_norm2')

                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
                ## --> [9, 9, 64]

            with tf.name_scope("conv3"):
                """
                Third convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                              filters=128,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv3")

                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name='batch_norm3')

                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
                ## --> [3, 3, 128]

            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                ## --> [1152]

            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="fc1")

            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs=self.fc,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=3,
                                              activation=None)

            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using 
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
