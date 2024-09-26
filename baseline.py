print("begin main script")
import gymnasium as gym
from stable_baselines3 import PPO
import envs

# env = gym.make("envs/SimpleWorld-v0", render_mode="human")
env = gym.make("envs/Balancing-v0")

# model = PPO("MlpPolicy", env, verbose=1)
model = PPO("MultiInputPolicy", env, verbose=1)

print("starting learning")
model.learn(total_timesteps=450_000)
print("finished learning")

env.toggle_on_vis()
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        obs = vec_env.reset()

env.close()
