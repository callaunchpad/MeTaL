import gym
import time
import numpy as np

env = gym.make('Ant-v2')
obs = env.reset()
while True:
    env.render()
    actions = np.random.randn(1, 8)
    env.step(actions)

    time.sleep(0.1)

