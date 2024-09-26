from gymnasium.envs.registration import register
from .grid_world import GridWorldEnv
from .simple_world import SimpleWorldEnv

register(
    id="envs/GridWorld-v0",
    entry_point="envs:GridWorldEnv",
    max_episode_steps=300,
)

register(
    id="envs/SimpleWorld-v0",
    entry_point="envs:SimpleWorldEnv",
    max_episode_steps=300,
)
