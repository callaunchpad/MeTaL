import numpy as np
from gym import utils
import environments.mujoco.tasks.mujoco_gym as mujoco_env
from gym.wrappers.time_limit import TimeLimit


class MoveTask(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'move_task.xml', 2)
        self.target_starting = [0.2, 0.2]
        self.target_speed = [0, -0.1]
        self.max_episode_steps = 400

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        self.goal = self.target_starting

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = self.target_speed
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])


if __name__ == "__main__":
    env2 = MoveTask()
    env = TimeLimit(env2, max_episode_steps=200)
    env.reset()
    while True:
        env.render()
        obs, _, _, _ = env.step([1, 1, -0.00000, -0.000])
        # print(obs)
