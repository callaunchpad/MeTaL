import environments.mujoco.env as env
import time
import numpy as np

env = env.make('move_task')
obs = env.reset()
while True:
    env.render()
    actions = np.random.randn(1, 8)
    env.step(actions)

    time.sleep(0.1)

