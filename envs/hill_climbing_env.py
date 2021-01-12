import numpy as np
import gym

from env_utils import cprint
from static_env import StaticEnv

''' Action Definitions '''
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


""" Reference: https://github.com/FelixOpolka/Single-Player-MCTS """
class HillClimbingEnv(gym.Env, StaticEnv):
    """
    Simple gym environment with the goal to navigate the player from its
    starting position to the highest point on a two-dimensional map within
    a limited number of steps. Rewards are defined as the difference in
    altitude between states minus a penalty for each step. The player starts
    in the lower left corner of the map and the highest point is in the upper
    right corner. Map layout mirrors CliffWalking environment:
    top left = (0, 0), top right = (0, m-1), bottom left = (n-1, 0),
    bottom right = (n-1, m-1).
    The setup of this environment was inspired by the energy landscape in
    protein folding.
    """

    altitudes = [[1, 1, 2, 3, 2, 4, 5],
                [3, 2, 2, 3, 3, 4, 4],
                [1, 1, 2, 2, 2, 3, 3],
                [0, 0, 0, 0, 1, 2, 2],
                [1, 2, 1, 0, 1, 3, 2],
                [0, 0, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0]]

    altitude_colors = ["green", "orange", "dark green",
                       "white", "black", "purple"]

    n_actions = 4

    def __init__(self):
        super(HillClimbingEnv, self).__init__()
        self.shape = (7, 7)
        self.ep_length = 15

        self.pos = (6, 0)
        self.step_idx = 0

    def reset(self):
        self.pos = (6, 0)
        self.step_idx = 0
        state = self.pos_to_state()
        return state, 0, False, None  # state, action, done, info

    def step(self, action):
        self.step_idx += 1
        alt_before = self.altitudes[self.pos[0]][self.pos[1]]
        state = self.next_state(self.pos_to_state(), action, self.shape)
        self.pos = np.unravel_index(state, self.shape)
        alt_after = self.altitudes[self.pos[0]][self.pos[1]]
        reward = alt_after - alt_before - 0.5
        done = self.is_done_state(state, self.step_idx, self.shape)

        return state, reward, done, None

    def render(self, mode='human'):
        if mode is not 'human':
            print(self.pos)
            return

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                end = " " if y < self.shape[0]-1 else ""
                bg_color = self.altitude_colors[self.altitudes[x][y]]
                color = "white" if bg_color == "black" else "black"
                if self.pos == (x, y):
                    cprint(" P ", "red", bg_color)
                else:
                    cprint(f" {self.altitudes[x][y]} ", color, bg_color)
            print()

    def next_state(self, state, action, shape=(7,7)):
        pos = np.unravel_index(state, shape)
        if action == UP:
            pos = (pos[0] - 1, pos[1])
        if action == DOWN:
            pos = (pos[0] + 1, pos[1])
        if action == LEFT:
            pos = (pos[0], pos[1] - 1)
        if action == RIGHT:
            pos = (pos[0], pos[1] + 1)
        pos = self._limit_coordinates(pos, shape)
        return pos[0] * shape[0] + pos[1]

    def is_done_state(self, state, step_idx, shape=(7,7)):
        return np.unravel_index(state, shape) == (0, 6) or step_idx >= self.ep_length

    def initial_state(self):
        return (self.shape[0]-1) * self.shape[0]

    def get_obs_for_states(self, states):
        return np.array(states)

    def get_return(self, state, step_idx, shape=(7, 7)):
        row, col = np.unravel(state, shape)
        return self.altitudes[row][col] - step_idx * 0.5

    def _limit_coordinates(self, coord, shape):
        coord = list(coord)
        coord[0] = min(coord[0], shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return tuple(coord)

    def pos_to_state(self):
        return self.pos[0] * self.shape[0] + self.pos[1]

if __name__ == '__main__':
    env = HillClimbingEnv()
    env.render()
    print(env.step(UP))
    env.render()
    print(env.step(RIGHT))
    env.render()
    print(env.step(DOWN))
    env.render()
    print(env.step(LEFT))
    env.render()


