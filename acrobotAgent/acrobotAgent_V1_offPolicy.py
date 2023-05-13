import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from IPython import display

env = gym.make("Acrobot-v1", render_mode='rgb_array')
observation, info = env.reset(seed=42)
# for _ in range(1):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
action = env.action_space
state = env.observation_space

dataPath = 'acrobotAgent/data/'
"""
the general agent shoudl be able to:
1. be able to analysis the env and understand the state space and action space


having the following functions:
1. defineNet(self)
2. trainFromExperience(self, filename)
3. generateExperience(self)
4. trainFromSimulation(self)

applied tricks or thoughts:
1. Replay trick, off-policy q-learning


future tricks or thoughts:
1. Coach and student network and other two network structure
2. Evaluation methods

future plan
1. construct MPC



"""


class Agent():
    def __init__(self, env, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, discount_factor=0.9,
                 num_of_episodes=500):
        # hyper parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.num_of_episodes = num_of_episodes
        self.env = env

        # get the shape of the state and action
        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape

        # define the model
        self.__buildd_model()
        pass

    def __buildd_model(self):
        self.net = torch.nn.Sequential(
            nn.Linear(self.state_shape[0], 25),
            nn.ReLU(),
            nn.Linear(25, self.env.action_space.n)
        )
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.net.apply(self._init_weights)
        pass

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)

    def generateExperience(self, num_experiences):

        experiences = []
        state, _ = self.env.reset()

        for i in range(num_experiences):
            # dont convert the data to other type, keep it as its original type and handle them when using them
            # this strategy can reduce confusion

            last_state = state
            action = self.env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)

            if type(state) == tuple:
                print("state is tuple")

            if done:
                # no need to know the order of the experiences
                state, _ = env.reset()
            else:
                experiences.append([last_state, action, reward, state, done])

        torch.save(experiences, dataPath + 'experiences_test.pt')
        pass

    def replay(self, episodes=100, batch_size=50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        print("Using device:", device)

        memory = torch.load(dataPath + 'experiences_test.pt')
        for i in range(episodes):
            batch = random.sample(memory, batch_size)

            for state, action, reward, next_state, done in batch:
                prediction = self.net(torch.tensor(state).to(device))
                target = prediction.clone()
                if not done:
                    target[action] = reward + self.discount_factor * torch.max(self.net(torch.tensor(next_state).to(device)))

                self.optimizer.zero_grad()
                self.loss(prediction, target).backward()
                self.optimizer.step()
            print("\repisode: ", i, " loss: ", self.loss(prediction, target).item(), flush=True)
        torch.save(self.net, dataPath + 'model_test.pt')

    def test(self, num_frames=100):
        model = torch.load(dataPath + 'model_test.pt')
        model.eval()
        state, _ = env.reset()
        frames = []

        import imageio

        for i in range(num_frames):
            action = model(torch.tensor(state)).argmax()
            state, reward, done, truncated, info = env.step(action)

            if done:
                state, _ = env.reset()
            img = env.render()
            frames.append(img)

        env.close()

        imageio.mimsave(dataPath + 'test.gif', frames, duration=1 / 60)

        pass


bot = Agent(env)

bot.generateExperience(100)

# bot.replay(episodes=2000, batch_size=1000)

bot.test(1000)
