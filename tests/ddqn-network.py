import gym
import numpy as np
import tensorflow as tf
import os


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = np.empty(shape=buffer_size, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buffer[self.index] = data
        self.length = min(self.length + 1, self.buffer_size)
        self.index = (self.index + 1) % self.buffer_size

    def get_length(self):
        return self.length

    def sample(self):
        batch = np.random.choice(self.buffer[:self.length], batch_size)
        batch_s = np.array([_[0] for _ in batch])
        batch_a = np.array([_[1] for _ in batch])
        batch_r = np.array([_[2] for _ in batch])
        batch_s_ = np.array([_[3] for _ in batch])
        batch_done = np.array([_[4] for _ in batch])

        return batch_s, batch_a, batch_r, batch_s_, batch_done


class ActorNetwork:
    def __init__(self, state_size, action_size, action_bound, batch_size, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau

        self.n_hidden1 = 400
        self.n_hidden2 = 300

        self.inputs, self.outputs, self.scaled, self.params = self._create_network("actor")
        self.t_inputs, self.t_outputs, self.t_scaled, self.t_params = self._create_network("target_actor")

        self.target_init = [self.t_params[i].assign(self.params[i]) for i in range(len(self.t_params))]
        self.update_target_params = \
            [self.t_params[i].assign(self.params[i] * self.tau +
                                     self.t_params[i] * (1. - self.tau)) for i in range(len(self.t_params))]
        self.critic_gradients = tf.placeholder(tf.float32, [None, action_size])
        self.actor_gradients = tf.gradients(self.scaled, self.params, -self.critic_gradients)
        self.norm_actor_grads = [i / self.batch_size for i in self.actor_gradients]

        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.norm_actor_grads, self.params))

    def _create_network(self, scope):
        with tf.variable_scope(scope):
            inputs = tf.placeholder(tf.float32, [None, self.state_size])
            hidden1_layer = tf.layers.dense(inputs, self.n_hidden1, activation=tf.nn.relu)
            hidden2_layer = tf.layers.dense(hidden1_layer, self.n_hidden2, activation=tf.nn.relu)
            outputs = tf.layers.dense(hidden2_layer, self.action_size, activation=tf.nn.tanh,
                                      kernel_initializer=tf.initializers.random_uniform(minval=-0.003, maxval=0.003))
            scaled_outputs = self.action_bound * outputs
        params = tf.trainable_variables(scope)
        return inputs, outputs, scaled_outputs, params

    def get_action(self, sess, inputs):
        return sess.run(self.scaled, feed_dict={
            self.inputs: inputs
        })

    def get_target_action(self, sess, inputs):
        return sess.run(self.t_scaled, feed_dict={
            self.t_inputs: inputs
        })

    def train(self, sess, inputs, critic_gradients):
        sess.run(self.train_op, feed_dict={
            self.inputs: inputs,
            self.critic_gradients: critic_gradients
        })

    def init_target(self, sess):
        sess.run(self.target_init)

    def update_target(self, sess):
        sess.run(self.update_target_params)


class CriticNetwork:
    def __init__(self, state_size, action_size, batch_size, lr, tau):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau

        self.n_hidden1 = 400
        self.n_hidden2 = 300

        self.inputs, self.actions, self.outputs, self.params = self._create_network("critic")
        self.t_inputs, self.t_actions, self.t_outputs, self.t_params = self._create_network("target_critic")

        self.target_init = [self.t_params[i].assign(self.params[i]) for i in range(len(self.t_params))]
        self.update_target_params = \
            [self.t_params[i].assign(self.params[i] * self.tau +
                                     self.t_params[i] * (1. - self.tau)) for i in range(len(self.t_params))]

        self.q_pred = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.losses.mean_squared_error(self.q_pred, self.outputs)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_grads = tf.gradients(self.outputs, self.actions)

    def _create_network(self, scope):
        with tf.variable_scope(scope):
            inputs = tf.placeholder(tf.float32, [None, self.state_size])
            actions = tf.placeholder(tf.float32, [None, self.action_size])
            hidden1_layer = tf.layers.dense(inputs, self.n_hidden1, activation=tf.nn.relu)

            # t_s = tf.layers.dense(hidden1_layer, self.n_hidden2)
            # t_a = tf.layers.dense(actions, self.n_hidden2)
            # hidden2_layer = t_s + t_a
            # hidden2_layer = tf.nn.relu(hidden2_layer)
            hidden2_layer = tf.layers.dense(tf.concat([hidden1_layer, actions], 1),
                                            self.n_hidden2, activation=tf.nn.relu)

            outputs = tf.layers.dense(hidden2_layer, 1,
                                      kernel_initializer=tf.initializers.random_uniform(minval=-0.003, maxval=0.003))
        params = tf.trainable_variables(scope)
        return inputs, actions, outputs, params

    def get_value(self, sess, inputs, actions):
        return sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.actions: actions
        })

    def get_target_value(self, sess, inputs, actions):
        return sess.run(self.t_outputs, feed_dict={
            self.t_inputs: inputs,
            self.t_actions: actions
        })

    def get_action_grads(self, sess, inputs, actions):
        return sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.actions: actions
        })

    def train(self, sess, inputs, actions, q_pred):
        sess.run(self.train_op, feed_dict={
            self.inputs: inputs,
            self.actions: actions,
            self.q_pred: q_pred
        })

    def init_target(self, sess):
        sess.run(self.target_init)

    def update_target(self, sess):
        sess.run(self.update_target_params)


env = gym.make('Swimmer-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

print(state_dim, action_dim, action_bound, env.action_space.low)

n_per_render = 2000
n_per_train = 1
n_episodes = 2000
batch_size = 64
gamma = 0.99
save_index = 0

memory = ReplayMemory(50000, batch_size)
actor = ActorNetwork(state_dim, action_dim, action_bound, batch_size, 0.00005, 0.001)
critic = CriticNetwork(state_dim, action_dim, batch_size, 0.0005, 0.001)
actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

with tf.Session() as session:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    checkpoint = "./run-" + str(save_index) + ".ckpt"
    if os.path.isfile(checkpoint + ".meta"):
        saver.restore(session, checkpoint)
    elif save_index != 0:
        raise Exception("Session data not found!!")

    actor.init_target(session)
    critic.init_target(session)

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        episode_length = 0

        while True:
            # if episode % n_per_render == 0:
            #     env.render()

            if episode > 20:
                action = actor.get_action(session, obs[np.newaxis, :])[0] + actor_noise()
                # print(action)
            else:
                action = env.action_space.sample()

            new_obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

            memory.append([obs, action, reward, new_obs, done])
            # print(reward)
            obs = new_obs

            if episode_length % n_per_train == 0 and episode > 20:
                mem_s, mem_a, mem_r, mem_s_, mem_done = memory.sample()
                target_actions = actor.get_target_action(session, mem_s_)
                target_q = critic.get_target_value(session, mem_s_, target_actions)

                q_y = []
                for i in range(len(target_q)):
                    if mem_done[i]:
                        q_y.append([mem_r[i]])
                    else:
                        q_y.append(target_q[i] * gamma + mem_r[i])
                critic.train(session, mem_s, mem_a, q_y)

                actor_pred = actor.get_action(session, mem_s)
                critic_grads = critic.get_action_grads(session, mem_s, actor_pred)[0]

                actor.train(session, mem_s, critic_grads)
                actor.update_target(session)
                critic.update_target(session)

            if done:
                print(episode, episode_length, total_reward, total_reward / episode_length)
                break

    saver.save(session, "./run-" + str(save_index + 1) + ".ckpt")
    while True:
        obs = env.reset()
        while True:
            env.render()
            action = actor.get_action(session, obs[np.newaxis, :])[0]
            new_obs, reward, done, info = env.step(action)
            obs = new_obs

            if done:
                break
