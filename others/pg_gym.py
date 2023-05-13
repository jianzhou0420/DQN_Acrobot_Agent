import gymnasium as gym

env = gym.make("Acrobot-v1")
observation, info = env.reset(seed=42)
# for _ in range(1):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
action = env.action_space.sample()
observation = env.observation_space.sample()

"""
the general agent
"""


class Agent():
    def __init__(self, env):
        self.env = env
        pass

    def defineNet(self):
        pass

    def trainFromExperience(self, filename):
        pass

    def generateExperience(self):
        pass

    def trainFromSimulation(self):
        pass
