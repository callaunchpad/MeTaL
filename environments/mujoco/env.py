from environments.mujoco.tasks.define_task import Task
from gym.wrappers.time_limit import TimeLimit

envs = {"simple_task": TimeLimit(Task("simple_task.xml"), max_episode_steps=1200),
        "speed_task": TimeLimit(Task("large_target.xml", speed_direction=1), max_episode_steps=1200),
        "slow_task": TimeLimit(Task("large_target.xml", speed_direction=-1), max_episode_steps=1500),
        "away_task": TimeLimit(Task("simple_task.xml", reward_scale=-1), max_episode_steps=600),
        }


def make(name):
    if name not in envs:
        raise Exception("Environment {} does not exist!".format(name))
    return envs[name]
