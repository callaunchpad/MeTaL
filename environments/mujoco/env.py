from environments.mujoco.tasks.speed_task import SpeedTask
from environments.mujoco.tasks.simple_task import SimpleTask
from environments.mujoco.tasks.slow_task import SlowTask
from gym.wrappers.time_limit import TimeLimit

envs = {"simple_task": TimeLimit(SimpleTask(), max_episode_steps=600),
        "speed_task": TimeLimit(SpeedTask(), max_episode_steps=600),
        "slow_task": TimeLimit(SlowTask(), max_episode_steps=600)}


def make(name):
    if name not in envs:
        raise Exception("Environment {} does not exist!".format(name))
    return envs[name]
