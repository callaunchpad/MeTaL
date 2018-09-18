import tensorflow as tf
from algorithms import architectures


def discount_and_normalize_rewards(episode_rewards):
    """
    Discounts and normalizes rewards from an episode

    TODO: Review and finish
    """

    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards


def SampleGaussian(means, log_simga_sqs):
    """
    Differentiably samples from Guassian using reparameterization trick.

    Args:
        means: Tensor of mean values
        log_sigma_sqs: Tensor of the logarithms of the variances
    Returs:
        Tensor of sampled gussian
    @Authors: Arsh Zahed
    """
    unit = tf.random_norm(means.shape, 0, 1)
    with_var = tf.sqrt(tf.exp(log_simga_sqs)) * eps
    return (with_var + means)


class PGFFNN:
    """Policy Gradient agent with only FFNN parameters

    Note: Minimal variance reduction, no natural gradient or trust-region optimization
    """

    def __init__(self, env, state_size, action_size, is_discrete, hparams):
        """
        Builds the graph for Feed Forward NN Policy Gradient agent 

        Args:
            state_size: Integer size of state Tensor
            action_size: Integer size of action Tensor
            is_discrete: Boolean, True if discrete space, False if continuous
            hparams: Dictionary of hyperparameters
                'learning_rate': Learning rate
                'output_size': Dimensionality of output
                'hidden_sizes': List of hidden layer sizes
                'activations': List of activation functions for each layer
        Returns:
            Output tensor of shape [None, output_size]
        @Authors: Arsh Zahed
        """
        self.env = env

        self.input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
        self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
        self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], 
                                                     name="discounted_episode_rewards")

        self.mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")
        pre_distr = architectures.FeedForward(self.input_, hparams, name='ffn_policygrad')

        if is_discrete:
            self.action_distribution = tf.nn.softmax(pre_distr)

            with tf.name_scope("loss"):
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.action_distribution, 
                                                                          labels = self.actions)
                self.loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_) 
                tf.summary.scalar('loss', self.loss)
        else:
            # TODO
            pass    

        with tf.name_scope("train"):
            self.train_opt = tf.train.AdamOptimizer(hparams['learning_rate']).minimize(self.loss)


    def train(self, sess, num_ep ):
        for episode in range(num_ep):
        
            episode_rewards_sum = 0

            # Launch the game
            state = self.env.reset()
            
            self.env.render()
               
            while True:
                # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, 
                # WE'RE OUTPUT PROBABILITIES.
                action_probability_distribution = sess.run(self.action_distribution, 
                                                           feed_dict={self.input_: state.reshape([1,4])})
                # select action w.r.t the actions prob
                action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                          p=action_probability_distribution.ravel())

                # Perform a
                new_state, reward, done, info = self.env.step(action)
                # Store s, a, r
                episode_states.append(state)

                # For actions because we output only one (the index) 
                # we need 2 (1 is for the action taken)
                action_ = np.zeros(action_size)
                action_[action] = 1
                
                episode_actions.append(action_)
                episode_rewards.append(reward)

                if done:
                    # Calculate sum reward
                    episode_rewards_sum = np.sum(episode_rewards)
                    allRewards.append(episode_rewards_sum)
                    total_rewards = np.sum(allRewards)
                    # Mean reward
                    mean_reward = np.divide(total_rewards, episode+1)
                    maximumRewardRecorded = np.amax(allRewards)
                    print("==========================================")
                    print("Episode: ", episode)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward", mean_reward)
                    print("Max reward so far: ", maximumRewardRecorded)

                    # Calculate discounted reward
                    discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

                    loss_, _ = sess.run(
                        [self.loss, self.train_opt],
                        feed_dict={self.input_: np.vstack(np.array(episode_states)),
                        self.actions: np.vstack(np.array(episode_actions)),
                        self.discounted_episode_rewards_: discounted_episode_rewards}
                        )



class PGLSTM:
    """
    TODO
    """

    def __init__(self):
        return

 

class PGConvNetwork:
    """
    A basic network that performs convolutions and 
    """
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_= tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")
            
                
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
                self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                             filters = 32,
                                             kernel_size = [8,8],
                                             strides = [4,4],
                                             padding = "VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             name = "conv1")

                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm1')

                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
                ## --> [20, 20, 32]
            
            with tf.name_scope("conv2"):
                """
                Second convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                     filters = 64,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv2")

                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm2')

                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
                ## --> [9, 9, 64]
            
            with tf.name_scope("conv3"):
                """
                Third convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                     filters = 128,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv3")

                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm3')

                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
                ## --> [3, 3, 128]
            
            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                ## --> [1152]
            
            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs = self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="fc1")
            
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs = self.fc, 
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units = 3, 
                                            activation=None)
            
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)
                

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using 
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_) 
        
    
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)