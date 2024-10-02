from gymnasium.envs.registration import register
from .grid_world import GridWorldEnv
from .simple_world import SimpleWorldEnv
from .simple_waypoints import SimpleWaypointsEnv
from .rocket_balancing import RocketBalancingEnv
from .rocket_waypoints import RocketWaypointsEnv

register(
    id="envs/GridWorld-v0",
    entry_point="envs:GridWorldEnv",
    max_episode_steps=300,
)

register(
    id="envs/SimpleWorld-v0",
    entry_point="envs:SimpleWorldEnv",
    max_episode_steps=600,
)

register(
    id="envs/SimpleWaypoints-v0",
    entry_point="envs:SimpleWaypointsEnv",
    max_episode_steps=600,
)

register(
    id="envs/RocketBalancing-v0",
    entry_point="envs:RocketBalancingEnv",
    max_episode_steps=1200,
)

register(
    id="envs/RocketWaypoints-v0",
    entry_point="envs:RocketWaypointsEnv",
    max_episode_steps=1200,
)
