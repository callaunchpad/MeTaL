import gym
import numpy as np
import tensorflow as tf
from algorithms.policygrad import PGFFNetwork

n_max_iter = 1500
n_games = 1500
discount_rate = 0.99

env = gym.make('CartPole-v0')
env._max_episode_steps = n_max_iter
env_obs_n = 4
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

    discounted_rewards = []
    accumulated_reward = 0
    for step in reversed(range(len(rewards))):
        accumulated_reward = rewards[step] + accumulated_reward * discount_rate
        discounted_rewards.insert(0, accumulated_reward)
    rewards_mean = np.mean(discounted_rewards)
    rewards_std = np.std(discounted_rewards)
    discounted_rewards = [(reward - rewards_mean) / rewards_std for reward in discounted_rewards]

    actions = np.expand_dims(actions, axis=1)
    discounted_rewards = np.expand_dims(discounted_rewards, axis=1)

    error = agent.train(states, actions, discounted_rewards)
    print("Game: {}, Error: {}, Game Length: {}, Total Reward: {}".format(game, error, len(actions), sum(rewards)))
