"""
Trains PGFFNetwork using meta-learning and then tests it on a normal environment

@Authors: Andrew Dickson
"""

import numpy as np
import tensorflow as tf
import gym
import os
import sys
import random

from tensorflow import flags
from tensorflow import app

from reptile import Reptile

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(dir_path))

from algorithms.policygrad import PGFFNetwork

flags.DEFINE_string('save_dir', '/tmp/train_log', 'Directory to checkpoint to.')
flags.DEFINE_integer('max_episodes', 1500, 'Maximum number of episodes.')
flags.DEFINE_float('learning_rate', .01, 'Learning rate.')
flags.DEFINE_float('discount', .95, 'MDP discount rate.')
flags.DEFINE_integer('env_act_n', 2, 'Number of actions in environment')
flags.DEFINE_integer('env_obs_n', 4, 'Number of observations in environment')
flags.DEFINE_integer('single_task_steps', 5, 'Number of times to repeat a given task in reptile training')
flags.DEFINE_integer('steps', 21, 'Number of times to sample tasks. Total steps is steps * single_task_steps')
flags.DEFINE_float('meta_step_size', 0.1, 'Weight to give to each reptile update')

FLAGS = flags.FLAGS


def main(argv):
	del argv  # Unused

	n_max_iter = FLAGS.max_episodes
	discount_rate = FLAGS.discount
	learning_rate = FLAGS.learning_rate

	#Create a set of tasks for meta-learning
	tasks = []
	for i in range(20):
		task = gym.make('CartPole-v0')
		task._max_episode_steps = n_max_iter
		task.unwrapped.gravity += random.random()*2-1
		task.unwrapped.length += 0.2*random.random()-0.1
		tasks.append(task)

	# Setup Gym Environment
	env = gym.make('CartPole-v0')
	env._max_episode_steps = n_max_iter

	# environment observation size
	env_obs_n = 4
	# environment action size
	env_act_n = 2

	#Create agent
	ff_hparams = {
		'hidden_sizes': [30, 30],
		'activations': [tf.nn.leaky_relu, tf.nn.leaky_relu],
		'output_size': env_act_n,
		'kernel_initializers': [tf.contrib.layers.xavier_initializer(),
								tf.contrib.layers.xavier_initializer(),
								tf.contrib.layers.xavier_initializer()]
	}

	agent = PGFFNetwork(env_obs_n, env_act_n, ff_hparams, learning_rate)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		saver = tf.train.Saver()
		# Do meta learning
		reptile = Reptile(sess, agent, tasks, FLAGS)
		reptile.train()

		# Test meta learning on original game
		for task in tasks:
			reptile.load()

			obs = task.reset()
			# store states, actions, and rewards
			states = []
			actions = []
			rewards = []
			for _ in range(n_max_iter):
				action_dist = agent.action_dist(obs[np.newaxis, :], sess)
				action = np.random.choice(np.arange(env_act_n), p=np.squeeze(action_dist))
				new_obs, reward, done, info = task.step(action)

				states.append(obs)
				actions.append(action)
				rewards.append(reward)
				obs = new_obs
				if done:
					break

			# discount rewards
			discounted_rewards = []
			accumulated_reward = 0
			for step in reversed(range(len(rewards))):
				accumulated_reward = rewards[step] + accumulated_reward * discount_rate
				discounted_rewards.insert(0, accumulated_reward)
			# normalize discounted rewards
			discounted_rewards -= np.mean(discounted_rewards)
			discounted_rewards /= np.std(discounted_rewards)

			# train agent
			error = agent.train(states, actions, discounted_rewards, sess)
			print("Gravity: {}, Length: {}, Error: {}, Game Length: {}, Total Reward: {}".format(task.unwrapped.gravity, task.unwrapped.length, error, len(actions), sum(rewards)))


if __name__ == "__main__":
	app.run(main)