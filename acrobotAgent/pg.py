import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

env = gym.make("Acrobot-v1")
observation, info = env.reset()
counter = 0
record_reward = []
experiences = []
for episode in range(1000):
    state, info = env.reset()
    for _ in range(10000):

        action = env.action_space.sample()  # this is where you would insert your policy
        new_state, reward, terminated, truncated, info = env.step(action)
        record_reward.append(reward)

        if reward == 0:
            counter += 1
            experiences.append([state, action, reward, new_state])
            print('yes')
        state = new_state
        if terminated or truncated:
            break
torch.save(experiences, 'success_experiences.pt')
env.close()
print(counter)
print(np.mean(record_reward))
