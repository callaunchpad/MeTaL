"""
Tests PGFFNetwork on an environment

@Authors: Yi Liu
"""

import numpy as np
import tensorflow as tf
import gym
import os
import sys

from tensorflow import flags
from tensorflow import app

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(dir_path))

from algorithms.naturalgrad import NGFFNetwork

flags.DEFINE_string('save_dir', '/tmp/train_log', 'Directory to checkpoint to.')
flags.DEFINE_integer('max_episodes', 150000, 'Maximum number of episodes.')
flags.DEFINE_integer('num_games', 1500, 'Number of games played')
flags.DEFINE_float('learning_rate', .01, 'Learning rate.')
flags.DEFINE_float('discount', .95, 'MDP discount rate.')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused

    n_max_iter = FLAGS.max_episodes
    n_games = FLAGS.num_games
    discount_rate = FLAGS.discount
    learning_rate = FLAGS.learning_rate

    # Setup Gym Environment
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

    agent = NGFFNetwork(env_obs_n, env_act_n, ff_hparams, learning_rate)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for game in range(n_games):
            obs = env.reset()
            # store states, actions, and rewards
            states = []
            actions = []
            rewards = []
            #action_dists = []
            for _ in range(n_max_iter):
                action_dist = agent.action_dist(obs[np.newaxis, :], sess)
            #    action_dists.append(action_dist)
                action = np.random.choice(np.arange(env_act_n), p=np.squeeze(action_dist))
                obs, reward, done, info = env.step(action)

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                if done:
                    break
            #Convert actions to one hot vectors
            
            #one_hot_actions = []
            #one_hot_matrix = np.eye(env_act_n)
            #for action in actions:
            #    one_hot_actions.append(one_hot_matrix[action])
            #actions = one_hot_actions
            

            #Convert list of action distributions to numpy array
            #action_dists = np.concatenate(action_dists, axis=0)
            #std_devs = np.zeros(action_dists.shape)+0.6


            # discount rewards
            discounted_rewards = []
            accumulated_reward = 0
            for step in reversed(range(len(rewards))):
                accumulated_reward = rewards[step] + accumulated_reward * discount_rate
                discounted_rewards.insert(0, accumulated_reward)
            # normalize discounted rewards
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            print(actions)
            # train agent
            error = agent.train(states, actions, discounted_rewards, sess)
            print("Game: {}, Error: {}, Game Length: {}, Total Reward: {}".format(game, error, len(actions), sum(rewards)))


if __name__ == "__main__":
    app.run(main)