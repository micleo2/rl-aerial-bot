import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


def limit_norm(vector, max_norm):
    """Limits the norm of a vector to a maximum value."""
    norm = np.linalg.norm(vector)
    if norm > max_norm:
        return vector * max_norm / norm
    else:
        return vector


class SimpleWaypointsEnv(gym.Env):
    """
    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | T1-Rel-x              | -window_size        | window_size       |
    | 1   | T1-Rel-y              | -window_size        | window_size       |
    | 2   | T2-Rel-x              | -window_size        | window_size       |
    | 3   | T2-Rel-y              | -window_size        | window_size       |
    | 4   | Velocity-x            | -max_vel            | max_vel           |
    | 5   | Velocity-y            | -max_vel            | max_vel           |

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=5):
        self.window_size = 1024  # The size of the PyGame window
        self.size = self.window_size
        self.square_size = 10
        self.win_distance = 20
        self.max_vel = 20

        # left, right, up, down
        self.action_space = spaces.Discrete(5)
        lows = np.array(
            [
                -self.size,
                -self.size,
                -self.size,
                -self.size,
                -self.max_vel,
                -self.max_vel,
            ]
        )
        highs = np.array(
            [
                self.size,
                self.size,
                self.size,
                self.size,
                self.max_vel,
                self.max_vel,
            ]
        )
        self.observation_space = spaces.Box(lows, highs, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def toggle_on_vis(self):
        self.render_mode = "human"

    def _get_obs(self):
        t1 = self._agent_pos - self._t1_pos
        t2 = self._agent_pos - self._t2_pos
        v = self._agent_vel
        return np.array([t1[0], t1[1], t2[0], t2[0], v[0], v[1]], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_pos = self.np_random.random(size=(2,)) * self.size / 4
        if self.np_random.random() > 0.5:
            self._agent_pos[0] = self.size - self._agent_pos[0]
        if self.np_random.random() > 0.5:
            self._agent_pos[1] = self.size - self._agent_pos[1]

        self._t1_pos = self.np_random.uniform(50, self.size - 50, size=(2,))
        self._t2_pos = self.np_random.uniform(50, self.size - 50, size=(2,))

        theta = self.np_random.random() * np.pi * 2
        vel_mag = 0.5
        self._agent_vel = (
            np.array(
                [
                    np.cos(theta),
                    np.sin(theta),
                ]
            )
            * vel_mag
        )
        self.timestep = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.timestep += 1
        dir = [0, 0]
        if action == 0:
            dir = [1, 0]
        elif action == 1:
            dir = [-1, 0]
        elif action == 2:
            dir = [0, 1]
        elif action == 3:
            dir = [0, -1]
        dir = np.array(dir)
        accel_mag = 0.5
        self._agent_vel = limit_norm(self._agent_vel + dir * accel_mag, self.max_vel)
        prev_dist = np.linalg.norm(self._agent_pos - self._t1_pos)
        self._agent_pos += self._agent_vel
        cur_dist = np.linalg.norm(self._agent_pos - self._t1_pos)

        reached = cur_dist < self.win_distance
        reached_reward = 2000 if reached else 0
        dist_reward = (prev_dist - cur_dist) / 20
        reward = reached_reward + dist_reward

        terminated = False
        if self._agent_pos[0] < 0 or self._agent_pos[0] >= self.size - 1:
            terminated = True
        if self._agent_pos[1] < 0 or self._agent_pos[1] >= self.size - 1:
            terminated = True
        if self.render_mode == "human":
            self._render_frame()

        if reached:
            self._t1_pos = self._t2_pos
            self._t2_pos = self.np_random.uniform(50, self.size - 50, size=(2,))

        obs = self._get_obs()
        inf = self._get_info()
        return obs, reward, terminated, False, inf

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        sz = self.square_size

        # T1
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            (int(self._t1_pos[0]), self._t1_pos[1], sz, sz),
        )
        # T2
        pygame.draw.rect(
            canvas,
            (255, 150, 0),
            (int(self._t2_pos[0]), self._t2_pos[1], sz, sz),
        )

        # agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            (
                int(self._agent_pos[0]),
                int(self._agent_pos[1]),
                sz,
                sz,
            ),
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
