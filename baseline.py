print("begin main script")
import gymnasium as gym
import pygame
from gymnasium.utils.play import play
from stable_baselines3 import PPO
import envs

# env = gym.make("envs/Balancing-v0", render_mode="human")
env = gym.make("envs/Balancing-v0")

# if True:
#     mapping = {(pygame.K_LEFT,): 1, (pygame.K_RIGHT,): 2, (pygame.K_SPACE,): 3}
#     env = gym.make("envs/Balancing-v0", render_mode="rgb_array")
#     play(env, keys_to_action=mapping)
#     exit()

model = PPO("MultiInputPolicy", env, verbose=1)

print("starting learning")
model.learn(total_timesteps=450_000)
print("finished learning")

env.toggle_on_vis()
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    # action, _states = model.predict(obs, deterministic=True)
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        obs = vec_env.reset()

env.close()
