import numpy as np
import tensorflow as tf
import gym
import os
import sys
import random

from tensorflow import flags
from tensorflow import app

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(dir_path))

from algorithms.policygrad import PGFFNetwork

flags.DEFINE_string('save_dir', '/tmp/train_log', 'Directory to checkpoint to.')
flags.DEFINE_integer('max_episodes', 150, 'Maximum number of episodes.')
flags.DEFINE_integer('num_games', 20, 'Number of games played')
flags.DEFINE_float('learning_rate', .01, 'Learning rate.')
flags.DEFINE_float('discount', .95, 'MDP discount rate.')

FLAGS = flags.FLAGS

actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

actions_list = [[1, 0], [0, 1], [-1, 0], [0, -1]]


def is_valid(state, maze):
    if(state[0] >= len(maze) or state[1] >= len(maze[0]) \
       or state[0] < 0 or state[1] < 0 or maze[state[0]][state[1]] == 1): 
        return False
    else:
        return True

def get_distance_map(maze, goal):
    distance_map = [[float("inf")] * len(maze[0]) for _ in maze]
    distance_map[goal[0]][goal[1]] = 0
    shell = [goal]
    while shell:
        new_shell = []
        for state in shell: 
            distance = distance_map[state[0]][state[1]]
            for action in actions:
                new_state = (state + action).tolist()
                if(is_valid(new_state, maze)):
                    if(distance_map[new_state[0]][new_state[1]] > (distance + 1)):
                        distance_map[new_state[0]][new_state[1]] = distance + 1
                        new_shell.append(new_state)
        shell = new_shell
    return distance_map  

def movement(state, action, maze, distance_map):
    next_state = state + action
    if(is_valid(next_state, maze)):
        return (state, action, distance_map[state[0]][state[1]] \
                - distance_map[next_state[0]][next_state[1]], np.array(next_state))
    else:
        return (state, action, 0, state)


def train_maze_agent(maze, goal):
    n_max_iter = FLAGS.max_episodes
    n_games = FLAGS.num_games
    discount_rate = FLAGS.discount
    learning_rate = FLAGS.learning_rate

    # environment observation size
    env_obs_n = 2
    # environment action size
    env_act_n = 4

    ff_hparams = {
        'hidden_sizes': [30, 30],
        'activations': [tf.nn.leaky_relu, tf.nn.leaky_relu],
        'output_size': env_act_n,
        'kernel_initializers': [tf.contrib.layers.xavier_initializer()] * 3
    }

    agent = PGFFNetwork(env_obs_n, env_act_n, ff_hparams, learning_rate)

    distance_map = get_distance_map(maze, goal)

    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        game_length = 150
        tf.summary_scalar("game_length", game_length)
        for game in range(n_games):
            obs = np.array([0, 0])
            # store states, actions, and rewards
            states = []
            actions = []
            rewards = []
            for _ in range(n_max_iter):
                action_dist = agent.action_dist(obs[np.newaxis, :], sess)
                action = np.random.choice(np.arange(env_act_n), p=np.squeeze(action_dist))
                _, _, reward, new_obs = movement(obs, actions_list[action], maze, distance_map)
                done = (new_obs[0] == goal[0] and new_obs[1] == goal[1])
                if(done):
                    reward += 5
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
            print("Game: {}, Error: {}, Game Length: {}, Total Reward: {}".format(game, error, len(actions), sum(discounted_rewards)))
            game_length = len(actions)

        print("finished training")

        def agent_distribution(state):
            return agent.action_dist(state[np.newaxis, :], sess)[0, :]

        return agent_distribution, sess
