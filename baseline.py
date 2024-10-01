print("begin main script")
import numpy as np
import gymnasium as gym
import pygame
from gymnasium.utils.play import play
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN, A2C
import envs

# env_name = "CartPole-v1"
env_name = "envs/Balancing-v0"
env = gym.make(env_name)

# if True:
#     mapping = {(pygame.K_LEFT,): 2, (pygame.K_RIGHT,): 3, (pygame.K_SPACE,): 1}
#     env = gym.make("envs/Balancing-v0", render_mode="rgb_array")
#     play(env, keys_to_action=mapping)
#     exit()

policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

print("starting learning")
model.learn(total_timesteps=1_000_000)
print("finished learning")

model.set_env(gym.make(env_name, render_mode="human"))
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
