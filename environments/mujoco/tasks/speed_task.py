import numpy as np
from gym import utils
import environments.mujoco.tasks.mujoco_gym as mujoco_env
from gym.wrappers.time_limit import TimeLimit


class SpeedTask(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'speed_task.xml', 2)

    def step(self, a):
        speed = a[0]
        assert 0 <= speed <= 1
        theta = self.sim.data.qpos.flat[2]
        action_vector = [speed * np.cos(theta), speed * np.sin(theta), a[1]]

        vec = self.get_body_com("agent") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward = reward_dist
        self.do_simulation(action_vector, self.frame_skip)
        ob = self._get_obs()
        done = False

        if np.linalg.norm(vec) <= 0.02:
            done = True
            reward = np.linalg.norm(self.sim.data.qvel.flat[:2])
        return ob, reward, done, dict(reward_dist=reward_dist)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
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
            self.sim.data.qvel.flat[:2],
            self.get_body_com("agent")[0:2] - self.get_body_com("target")[0:2]
        ])


if __name__ == "__main__":
    env2 = SpeedTask()
    env = TimeLimit(env2, max_episode_steps=50000)
    env.reset()
    while True:
        env.render()
        obs, reward, _, _ = env.step([0.2, 0.001])
        # print(reward)
