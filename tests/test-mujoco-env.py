import environments.mujoco.env as env

env = env.make('speed_task')
obs = env.reset()
while True:
    env.render()
    env.step([1, 0.001])

