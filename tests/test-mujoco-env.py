import environments.mujoco.env as mujoco_env

env = mujoco_env.make('speed_task')
obs = env.reset()
while True:
    env.render()
    obs, _, _, _ = env.step([1, 0.001])
    print(obs)

