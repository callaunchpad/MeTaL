import numpy as np

class PointAgent():
    import collections

    def __init__(self, goal, maxdist):
        self.goal = goal
        self.loc = [0, 0]
        self.maxdist = maxdist

    def dist_to_goal(self, loc):
        return ((loc[0] - self.goal[0]) ** 2 + (loc[1] - self.goal[1]) ** 2) ** 0.5

    def get_action_distr(self, loc):
        dist = min(self.maxdist, self.dist_to_goal(loc))
        angle = np.arctan((loc[1] - self.goal[1]) / (loc[0] - self.goal[0]))
        return (dist, angle)
    
    def move(self, dist_distr, angle_distr):
        distance = dist_distr()
        angle = angle_distr()
        self.loc[0] += np.cos(angle) * distance
        self.loc[1] += np.sin(angle) * distance

class NoisyPointAgent(PointAgent):
    def __init__(self, goal, maxdist, dist_noise, angle_noise):
        self.dist_noise = dist_noise
        self.angle_noise = angle_noise
        PointAgent.__init__(self, goal, maxdist)
        
    def get_action_distr(self, loc):
        no_noise = list(PointAgent.get_action_distr(self, loc))
        print(no_noise)
        noisy = no_noise
        noisy[0] *= self.dist_noise()
        noisy[1] += self.angle_noise()
        return noisy

if __name__ == '__main__':
    agent = NoisyPointAgent(goal=(10,10), maxdist=5, dist_noise = lambda: np.random.normal(loc=1), angle_noise = lambda: np.random.normal(scale=0.1))
    print(agent.get_action_distr(loc=(0,0)))
