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
n_games = 1500
# number of episodes to run for each game
n_episodes = 100
discount_rate = 0.96

env = gym.make('CartPole-v0')
env._max_episode_steps = n_max_iter
# environment observation size
env_obs_n = 4
# environment action size
env_act_n = 2

ff_hparams = {
    'hidden_sizes': [30, 30],
    'activations': [tf.nn.leaky_relu, tf.nn.leaky_relu],
    'output_size': env_act_n,
    'kernel_initializers': [tf.contrib.layers.xavier_initializer(),
                            tf.contrib.layers.xavier_initializer(),
                            tf.contrib.layers.xavier_initializer()]
}

learning_rate = 0.004
sess = tf.InteractiveSession()
agent = PGFFNetwork(sess, env_obs_n, env_act_n, ff_hparams, learning_rate, n_episodes)
tf.global_variables_initializer().run()

for game in range(n_games):
    # store states, actions, and rewards
    states = []
    actions = []
    rewards = []
    discounted_rewards = []
    for episode in range(agent.n_episodes):
        states.append([])
        actions.append([])
        rewards.append([])
        discounted_rewards.append([])
        obs = env.reset()
        for _ in range(n_max_iter):
            action_dist = agent.action_dist(obs[np.newaxis, :])
            action = np.random.choice(np.arange(env_act_n), p=np.squeeze(action_dist))
            obs, reward, done, info = env.step(action)

            states[-1].append(obs)
            actions[-1].append(action)
            rewards[-1].append(reward)
            if done:
                break

        # discount rewards
        accumulated_reward = 0
        for step in reversed(range(len(rewards))):
            accumulated_reward = rewards[step] + accumulated_reward * discount_rate
            discounted_rewards[-1].insert(0, accumulated_reward)
    # normalize discounted rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    # train agent
    error = agent.train(states, actions, discounted_rewards)
    print("Game: {}, Error: {}, Game Length: {}, Total Reward: {}".format(game, error, len(actions), sum(rewards)))
