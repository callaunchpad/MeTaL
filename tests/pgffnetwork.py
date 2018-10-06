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

from algorithms.policygrad import PGFFNetwork

flags.DEFINE_string('save_dir', '/tmp/train_log', 'Directory to checkpoint to.')
flags.DEFINE_integer('max_episodes', 150000, 'Maximum number of episodes.')
flags.DEFINE_integer('num_games', 1500, 'Number of games played')
flags.DEFINE_float('learning_rate', .01, 'Learning rate.')
flags.DEFINE_float('discount', .95, 'MDP discount rate.')
flags.DEFINE_integer('num_environments', 5, 'Number of Environments to sample.')

FLAGS = flags.FLAGS

def main(argv):
    del argv  # Unused

    n_max_iter = FLAGS.max_episodes
    n_games = FLAGS.num_games
    discount_rate = FLAGS.discount
    learning_rate = FLAGS.learning_rate

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
    agent = PGFFNetwork(sess, env_obs_n, env_act_n, ff_hparams, learning_rate, n_environments, 0)
    tf.global_variables_initializer().run()


    envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
    for env in envs:
        env._max_episode_steps = n_max_iter

    for game in range(n_games):
        # store states, actions, and rewards
        observations = np.array([env.reset() for env in envs])
        states = [[]] * n_environments
        actions = [[]] * n_environments
        rewards = [[]] * n_environments
        dones = [None] * n_environments
        discounted_rewards = []

        for _ in range(n_max_iter):
            action_dist = agent.action_dist(observations[:, np.newaxis, :])
           
            for env_index, env in enumerate(envs):
                if not dones[env_index]:    
                    action = np.random.choice(np.arange(env_act_n), p=np.squeeze(action_dist[env_index]))
                    obs, reward, done, info = env.step(action)
                    obs = obs if not done else env.reset()
                    states[env_index].append(obs)
                    actions[env_index].append(action)
                    rewards[env_index].append(reward)
                    dones[env_index] = done
                else:
                    states[env_index].append(states[env_index][-1])
                    actions[env_index].append(actions[env_index][-1])
                    rewards[env_index].append(rewards[env_index][-1])
            
            # This probably needs to change?
            if all(dones):
                break

        agregated_rewards = []
        # discount rewards
        for env_index, env in enumerate(envs):
            discounted_rewards = []
            accumulated_reward = 0
            for step in reversed(range(len(rewards[env_index]))):
                accumulated_reward = rewards[env_index][step] + accumulated_reward * discount_rate
                discounted_rewards.insert(0, accumulated_reward)
            # normalize discounted rewards
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            agregated_rewards.append(discounted_rewards)

        # train agent
        error = agent.train(states, actions, agregated_rewards)
        avg_error = np.mean(error)
        sum_rewards = np.sum(rewards)
        print("Game: {}, Average Error: {}, Total Reward: {}".format(game, avg_error, sum_rewards))
        #print("Game: {}, Average Error: {}, Game Length: {}, Total Reward: {}".format(game, avg_error, len(actions), sum(rewards)))

if __name__ == "__main__":
    app.run(main)
