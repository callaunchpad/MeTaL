import numpy as np
from gym import utils
import environments.mujoco.tasks.mujoco_gym as mujoco_env


class Task(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, model_file, speed_direction=False, reward_scale=1):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_file, 2)
        self.speed_direction = speed_direction
        self.reward_scale = reward_scale

        self.distance = 0.04

    def step(self, a):
        speed = a[0] * 0.002
        rotation = a[1] * 0.02
        if speed > 1:
            speed = 1
        if speed < -1:
            speed = -1
        theta = self.sim.data.qpos.flat[2]
        action_vector = [speed * np.cos(theta), speed * np.sin(theta), rotation]

        vec = self.get_body_com("agent") - self.get_body_com("target")
        reward_dist = -4 * self.reward_scale * np.linalg.norm(vec)
        reward = reward_dist
        self.do_simulation(action_vector, self.frame_skip)
        ob = self._get_obs()
        done = False

        if self.speed_direction and np.linalg.norm(vec) <= self.distance:
            done = True
            reward = 28 * self.speed_direction * np.linalg.norm(self.sim.data.qvel.flat[:2])
        return ob, reward, done, dict(reward_dist=reward_dist)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq)
        while np.linalg.norm(qpos[0:2] - qpos[3:5]) <= self.distance:
            qpos = self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq)

        qpos[2] = self.np_random.uniform(low=-3, high=3, size=1)
        qvel = np.array([0, 0, 0, 0, 0])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[2]
        return np.concatenate([
            [np.cos(theta)],
            [np.sin(theta)],
            self.sim.data.qvel.flat[:3],
            self.get_body_com("agent")[0:2],
            self.get_body_com("agent")[0:2] - self.get_body_com("target")[0:2]
        ])
