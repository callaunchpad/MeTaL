#!/usr/bin/env python3

import math
import numpy as np
import matplotlib.pyplot as plt
class PointMassTask:
  action_size = 2
  state_size = 4
  def __init__(self, goal_position=(5,5), mass=0, initial_position=(0,0), initial_momentum=0, initial_direction=0, threshold_distance=0.01):
    """
    Creates point-mass task
    args:
        goal_position: tuple representing goal location
        mass: number representing mass of point
        initial_position: tuple representing initial position
        initial_momentum: magnitude of initial momentum; represents initial speed if mass is 0
        initial_direction: initial direction in radians
        threshold_distance: threshold distance from goal location for task completion
    """
    self.threshold_distance = threshold_distance
    self.goal_position = np.array(goal_position, dtype=np.float32)
    self.initial_position = np.array(initial_position, dtype=np.float32)
    self.initial_momentum = initial_momentum # actually represents initial speed if 0 mass
    self.mass = mass
    self.initial_direction = initial_direction
    self.reset()
    plt.ion()
  def step(self, action):
    """
    Returns new state and reward from action. 
    args:
        action: tuple of two numbers representing impulse magnitude and direction (radians); if mass is 0, the first value represents the new velocity
    """
    if isinstance(action, (int, np.integer)):
      raise ValueError('Action must be a tuple of two real-valued numbers representing impulse magnitude and direction. ')
    if self.mass == 0:
      self.velocity = to_cartesian(action[0], action[1]) # first input is new speed, second is direciton
    else:
      self.velocity += to_cartesian(action[0]/self.mass, action[1]) # first input is impulse magnitude, second is direciton
    self.position += self.velocity

    done = dist(self.position, self.goal_position) < self.threshold_distance
    return np.hstack([self.position, self.velocity]), 100 if done else 0, done, None # new_obs, reward, done, info

  def reset(self):
    if self.mass == 0:
      self.velocity = to_cartesian(self.initial_momentum, self.initial_direction)
    else:
      self.velocity = to_cartesian(self.initial_momentum/self.mass, self.initial_direction)
    self.position = self.initial_position

    return np.hstack([self.position, self.velocity])

  def render(self):
    xs = [self.position[0], self.goal_position[0]]
    ys = [self.position[1], self.goal_position[1]]
    colors = [(1,0,0), (0,0,0)]
    pts = plt.scatter(xs, ys, c=colors)
    
    plt.draw()
    plt.pause(0.2)
    plt.show()
    pts.remove()

def to_cartesian(magnitude, direction):
  return np.array([math.cos(direction)*magnitude, math.sin(direction)*magnitude])

def dist(p1, p2):
  return np.linalg.norm(p1-p2)


if __name__ == '__main__':
  task = PointMassTask(threshold_distance=1)
  import time
  done = False
  task.render()
  while not done:
    obs, reward, done, info = task.step([0.2, math.pi/4])
    print(task.position, task.velocity)
    task.render()
    time.sleep(0.2)
