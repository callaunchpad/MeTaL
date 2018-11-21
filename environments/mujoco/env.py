from environments.mujoco.tasks.move_task import MoveTask
from environments.mujoco.tasks.simple_task import SimpleTask
from gym.wrappers.time_limit import TimeLimit

envs = {"move_task": TimeLimit(MoveTask(), max_episode_steps=200),
        "simple_task": TimeLimit(SimpleTask(), max_episode_steps=1200)}


def make(name):
    if name not in envs:
        raise Exception("Environment {} does not exist!".format(name))
    return envs[name]
