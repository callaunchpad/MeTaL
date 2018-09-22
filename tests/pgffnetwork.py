"""
Tests PGFFNetwork on an environment

@Authors: Yi Liu
"""

import gym
import numpy as np
import tensorflow as tf
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(dir_path))

from algorithms.policygrad import PGFFNetwork

# maximum number of iterations of environment
n_max_iter = 150000
# number of games played
n_games = 15000
discount_rate = 0.99

env = gym.make('CartPole-v0')
env._max_episode_steps = n_max_iter
# environment observation size
env_obs_n = 4
# environment action size
env_act_n = 2

ff_hparams = {
    'hidden_sizes': [30, 30],
    'activations': [tf.nn.relu, tf.nn.relu],
    'output_size': env_act_n
}
learning_rate = 0.001
sess = tf.InteractiveSession()
agent = PGFFNetwork(sess, env_obs_n, env_act_n, ff_hparams, learning_rate)
tf.global_variables_initializer().run()

for game in range(n_games):
    obs = env.reset()
    # store states, actions, and rewards
    states = []
    actions = []
    rewards = []
    for _ in range(n_max_iter):
        action_dist = agent.action_dist(obs[np.newaxis, :])
        action = np.random.choice(np.arange(env_act_n), p=np.squeeze(action_dist))
        obs, reward, done, info = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    # discount rewards
    discounted_rewards = []
    accumulated_reward = 0
    for step in reversed(range(len(rewards))):
        accumulated_reward = rewards[step] + accumulated_reward * discount_rate
        discounted_rewards.insert(0, accumulated_reward)
    # normalize discounted rewards
    rewards_mean = np.mean(discounted_rewards)
    rewards_std = np.std(discounted_rewards)
    discounted_rewards = [(reward - rewards_mean) / rewards_std for reward in discounted_rewards]

    # format actions and rewards to proper dimensions
    actions = np.expand_dims(actions, axis=1)
    discounted_rewards = np.expand_dims(discounted_rewards, axis=1)

    # train agent
    error = agent.train(states, actions, discounted_rewards)
    print("Game: {}, Error: {}, Game Length: {}, Total Reward: {}".format(game, error, len(actions), sum(rewards)))


    // Build Computational Graph
    // None dimension indicates variable size
    // In this case, batch size
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    y_hat = NeuralNet(x)
    y_truth = tf.placeholder(dtype=tf.float32,
                             shape=[None, output_size])
    // Add loss to graph
    loss = LossFunction(y_hat, y_truth)
    opt = Optimizer(learning_rate, loss)

    with tf.Session() as sess:

        // Collect data
        input_, truth = GetData(...)

        // Associate values
        feed_dict = {x: input_, y_truth: truth}
        // will run up to optimizer
        sess.run(opt, feed_dict=feed_dict)


// Build Computation Graph

// Initialize Variables

with tf.Session() as sess:
    for i in range(num_iterations):
        // Get Batch of Data
        
        // Feed Data, Run session, 
        // Update Parameters

