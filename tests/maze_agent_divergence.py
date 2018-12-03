import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from maze_agent import train_maze_agent
import random


maze = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
 [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
 [1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
 [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

actions_list = [[1, 0], [0, 1], [-1, 0], [0, -1]]
actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

def is_valid(state, maze):
    if(state[0] >= len(maze) or state[1] >= len(maze[0]) \
       or state[0] < 0 or state[1] < 0 or maze[state[0]][state[1]] == 1): 
        return False
    else:
        return True

def get_distance_map(maze, goal):
    distance_map = [[float("inf")] * len(maze[0]) for _ in maze]
    distance_map[goal[0]][goal[1]] = 0
    shell = [goal]
    while shell:
        new_shell = []
        for state in shell: 
            distance = distance_map[state[0]][state[1]]
            for action in actions:
                new_state = (state + action).tolist()
                if(is_valid(new_state, maze)):
                    if(distance_map[new_state[0]][new_state[1]] > (distance + 1)):
                        distance_map[new_state[0]][new_state[1]] = distance + 1
                        new_shell.append(new_state)
        shell = new_shell
    return distance_map  

def movement(state, action, maze, distance_map):
    next_state = state + action
    if(is_valid(next_state, maze)):
        return (state, action, distance_map[state[0]][state[1]] \
                - distance_map[next_state[0]][next_state[1]], np.array(next_state))
    else:
        return (state, action, 0, state)

def random_movement(weights):
    total = sum(weights)
    rand_val = total*random.random()
    total = 0
    ret_val = 0
    for weight in weights:
        total += weight
        if(total > rand_val):
            return ret_val
        else:
            ret_val += 1

def random_agent(state):
    return np.array([0.2, 0.4, 0.1, 0.3])

def create_heatmap(agent, maze, start, num_walks, walk_len, goal):
    square_counts = maze*0
    distance_map = get_distance_map(maze, goal)
    
    for i in range(num_walks):
        state = start
        for j in range(walk_len):
            square_counts[state[0]][state[1]] += 1
            distribution = agent(state)
            action_num = random_movement(distribution)
            action = actions[action_num, :]
            _, _, _, next_state = movement(state, action, maze, distance_map)
            if(next_state[0] == goal[0] and next_state[1] == goal[1]):
                continue
            state = next_state
    max_count = np.amax(square_counts)
    return square_counts / max_count

def DiscreteJSD(P, Q):
    M = (P + Q)/2
    return (entropy(P, M) + entropy(Q, M))/2
        
def get_divergence(agent1, agent2, goal1, goal2, maze, start, num_walks, walk_len):
    distance_map = get_distance_map(maze, goal1)
    averages = []
    for i in range(num_walks):
        state1 = start
        state2 = start
        divergences = []
        for j in range(walk_len):
            p1 = agent1(state1)
            q1 = agent2(state1)
            p2 = agent1(state2)
            q2 = agent2(state2)
            action1 = actions[random_movement(p1), :]
            action2 = actions[random_movement(q2), :]
            divergences.append(((DiscreteJSD(p1, q1) ** (.5)) + (DiscreteJSD(p2, q2) ** (.5)))/2)
            _, _, _, state1 = movement(state1, action1, maze, distance_map)
            _, _, _, state2 = movement(state2, action2, maze, distance_map)
            if(state1[0] == goal1[0] and state1[1] == goal1[1]):
                continue
            if(state2[0] == goal2[0] and state2[1] == goal2[1]):
                continue
        averages.append(sum(divergences)/len(divergences))
    return sum(averages)/len(averages)

def main(argv):
    del argv  # Unused)
    agent_goals = [[9, 9], [0, 9]]
    agents = []
    for goal in agent_goals:
        graph = tf.Graph()
        with graph.as_default():
            agent = train_maze_agent(maze, goal)

        agents.append(agent)
    goal1 = agent_goals[0]
    goal2 = agent_goals[1]
    agent1 = agents[0]
    agent2 = agents[1]
    start = np.array([0, 0])
    num_walks = 100
    walk_len = 35
    print(get_divergence(agent1, agent1, goal1, goal1, maze, start, num_walks, walk_len))
    print(get_divergence(agent1, agent2, goal1, goal2, maze, start, num_walks, walk_len))
    print(get_divergence(agent2, agent2, goal2, goal2, maze, start, num_walks, walk_len))

def gen_heatmap():
    start = np.array([0, 0])
    num_walks = 100
    walk_len = 35

    baseline_agent = train_maze_agent(maze, [9, 9])
    baseline_goal = [9, 9]

    agents = []
    heatmap = np.zeros((10, 10))
    for i in range(9):
        for j in range(9):
            goal = [i, j]
            graph = tf.Graph()
            with graph.as_default():
                agent = train_maze_agent(maze, goal)
            divergence = get_divergence(baseline_agent, agent, baseline_goal, 
                goal, maze, start, num_walks, walk_len)
    return heatmap


if __name__ == "__main__":
    main(None)