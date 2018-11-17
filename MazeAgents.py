import dijkstra
import numpy as np

class MazeAgent():
    import collections

    def __init__(self, goal, walls, size):
        self.size = size
        self.goal = goal
        self.loc = np.array([0,0])
        self.graph = self.make_graph(walls)
        self.ACTIONS = [0, 1, 2, 3, 4]

    def get_action_distr(self, loc):
        distr = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        best_action = self.find_move_from_state(start=loc, state=self.find_optimal_first_move(loc))
        distr[best_action] = 1.0
        return distr
    
    # TODO: make states an enum
    def find_move_from_state(self, start, state):
        #make sure we only differ in one position by one move
        #NOTE! != is xor for booleans in python
        assert bool((start[0]-state[0])**2 == 1) != bool((start[1]-state[1])**2 == 1)
        if state[1] == start[1] - 1:
            return 0 # up
        elif state[1] == start[1] + 1:
            return 1 # down
        elif state[0] == start[0] + 1:
            return 2 # right
        elif state[0] == start[0] - 1:
            return 3 # left
        else:
            raise ValueError("bad states")
    
    def find_optimal_first_move(self, start):
        _, paths_back = self.graph.dijkstra(start)
        # starting with a dictionary of nodes to adjacent nodes closest to start, walk back to find best first move
        second_state = self.goal
        last_state = paths_back[self.goal]
        while last_state is not start:
            second_state = last_state
            last_state = paths_back[last_state]
        return second_state
    
    def make_graph(self, walls):
        x_coords = list(range(self.size))
        y_coords = list(range(self.size))
        states = []
        for start_x in x_coords:
            for start_y in y_coords:
                states.append((start_x, start_y))
                
        graph = dijkstra.Digraph(nodes=states)
        for start_x in x_coords:
            for start_y in y_coords:
                start = (start_x, start_y)
                left = (start_x-1, start_y)
                right = (start_x+1, start_y)
                up = (start_x, start_y+1)
                down = (start_x, start_y-1)
                
                if start_x - 1 >= 0 and left not in walls:
                    graph.addEdge(start, left, 1)
                if start_x + 1 < self.size and right not in walls:
                    graph.addEdge(start, right, 1)
                if start_y - 1 >= 0 and down not in walls:
                    graph.addEdge(start, down, 1)
                if start_x + 1 < self.size and up not in walls:
                    graph.addEdge(start, up, 1)
        return graph
                
    
    def move(self, distr):
        action = np.random.choice(self.ACTIONS, 1, p=distr)[0]
        if action == 0:
            self.loc[1] += 1
        elif action == 1:
            self.loc[1] -= 1
        elif action == 2:
            self.loc[0] += 1
        else:
            self.loc[0] -= 1

class NoisyMazeAgent(MazeAgent):
    def __init__(self, goal, walls, size, opt_prob):
        self.opt_prob = opt_prob
        self.noise = (1-opt_prob)/4
        MazeAgent.__init__(self, goal, walls, size)
        
    def get_action_distr(self, loc):
        no_noise = list(MazeAgent.get_action_distr(self, loc))
        print(no_noise)
        noisy = no_noise
        for index, elem in enumerate(no_noise):
            if no_noise[index] == 1:
                noisy[index] = self.opt_prob
            else:
                noisy[index] = self.noise
        return noisy

if __name__ == '__main__':
    agent = NoisyMazeAgent(goal=(2,0), walls=[], size=3, opt_prob=0.9)
    print(agent.get_action_distr(loc=(0,0)))
