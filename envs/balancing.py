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
        self.max_vel = 20

        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(0, self.size, shape=(2,), dtype=np.float64),
                "vel": spaces.Box(
                    -self.max_vel, self.max_vel, shape=(2,), dtype=np.float64
                ),
            }
        )

        # # none, left, right, boost
        # self.action_space = spaces.Discrete(4)

        # left, right, up, down
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def toggle_on_vis(self):
        self.render_mode = "human"

    def _get_obs(self):
        p = self._agent_pos - self._target_pos
        v = self._agent_vel
        return {
            "pos": p,
            "vel": v,
            # "theta": t,
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

        self._agent_theta = self.np_random.random() * np.pi * 2
        vel_mag = 0.5
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
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # is_boost = False
        # turn_strength = np.pi / 100
        # if action == 1:
        #     self._agent_theta += turn_strength
        # elif action == 2:
        #     self._agent_theta -= turn_strength
        # elif action == 3:
        #     is_boost = True

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

        # tpi = np.pi * 2
        # self._agent_theta %= tpi
        # if self._agent_theta < 0:
        #     self.theta = tpi + self._agent_theta
        # accel_mag = 1
        # accel = (
        #     np.array([np.cos(self._agent_theta), np.sin(self._agent_theta)])
        #     * is_boost
        #     * accel_mag
        # )

        # self._agent_vel = limit_norm(self._agent_vel + accel, self.max_vel)
        prev_dist = np.linalg.norm(self._agent_pos - self._target_pos)
        self._agent_pos = np.clip(self._agent_pos + self._agent_vel, 10, self.size - 10)
        cur_dist = np.linalg.norm(self._agent_pos - self._target_pos)
        target = self.size / 2
        terminated = cur_dist < self.win_distance
        reached_reward = 200 if terminated else 0
        # reached_reward = ((600 - self.timestep) / 10) ** 2 + 200 if terminated else 0
        # dist_reward = (np.sqrt((self.size**2) * 2) - d) / self.size
        dist_reward = (prev_dist - cur_dist) / 20
        if self.render_mode == "human":
            print(f"reached_reward={reached_reward}\tdist_reward={dist_reward}")
        # reward -= self.timestep / 600
        reward = reached_reward + dist_reward

        if self.render_mode == "human":
            self._render_frame()

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

        # First we draw the target
        square_size = 10
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            (int(self._target_pos[0]), self._target_pos[1], square_size, square_size),
        )
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
