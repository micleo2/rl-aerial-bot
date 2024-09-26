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


class BalancingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=5):
        self.window_size = 1024  # The size of the PyGame window
        self.size = self.window_size
        self.win_distance = 10
        self.max_vel = 10
        self.tpi = np.pi * 2
        self.gravity = 0.175
        self.boost_accel = 0.4

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(0, self.size, shape=(2,), dtype=np.float64),
                "vel": spaces.Box(
                    -self.max_vel, self.max_vel, shape=(2,), dtype=np.float64
                ),
                "theta": spaces.Box(
                    -self.tpi * 4, self.tpi * 4, shape=(1,), dtype=np.float64
                ),
            }
        )

        # none, left, right, boost
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def toggle_on_vis(self):
        self.render_mode = "human"

    def _get_obs(self):
        p = self._agent_pos
        v = self._agent_vel
        t = np.array([self._agent_theta])
        return {
            "pos": p,
            "vel": v,
            "theta": t,
        }

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

        self._target_pos = self.np_random.uniform(50, self.size - 50, size=(2,))
        self._agent_pos = np.array([self.size / 2, self.size / 2])

        # self._agent_theta = self.np_random.random() * np.pi * 2
        self._agent_theta = -np.pi / 2
        rn = np.pi / 10
        self._agent_theta += self.np_random.uniform(-rn, rn)
        # vel_mag = 0.5
        vel_mag = 0
        self._agent_vel = (
            np.array(
                [
                    np.cos(self._agent_theta),
                    np.sin(self._agent_theta),
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
        # none, left, right, boost
        is_boost = False
        turn_strength = np.pi / 60
        if action == 1:
            is_boost = True
        # if action == 1:
        #     self._agent_theta += turn_strength
        # elif action == 2:
        #     self._agent_theta -= turn_strength
        # elif action == 3:
        #     is_boost = True

        self._agent_theta = -np.pi / 2
        accel = (
            np.array([np.cos(self._agent_theta), np.sin(self._agent_theta)])
            * is_boost
            * self.boost_accel
        )
        accel[1] += self.gravity

        self._agent_vel = limit_norm(self._agent_vel + accel, self.max_vel)
        self._agent_pos += self._agent_vel
        reward = (self.max_vel - np.linalg.norm(self._agent_vel)) ** 2

        terminated = False
        if self._agent_pos[0] < 0 or self._agent_pos[0] >= self.size - 1:
            terminated = True
        if self._agent_pos[1] < 0 or self._agent_pos[1] >= self.size - 1:
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, False, info

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

        square_size = 10

        # # target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     (int(self._target_pos[0]), self._target_pos[1], square_size, square_size),
        # )

        # agent
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            (
                int(self._agent_pos[0]),
                int(self._agent_pos[1]),
                square_size,
                square_size,
            ),
        )
        xd = np.cos(self._agent_theta) * 20
        yd = np.sin(self._agent_theta) * 20
        # agent direction
        pygame.draw.line(
            canvas,
            (0, 255, 0),
            (
                int(self._agent_pos[0]),
                int(self._agent_pos[1]),
            ),
            (
                int(self._agent_pos[0] + xd),
                int(self._agent_pos[1] + yd),
            ),
            width=3,
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
