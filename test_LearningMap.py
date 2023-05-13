### Provide your code here
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

from envZJA.SnakeGame import SnakeGame

"""
the SnakeAgent is a class that can be used to train a snake agent
It has the following methods:
train() 
test() 
generateExperience() 
loadExperience()


"""


class SnakeAgent():
    def __init__(self, env):
        # set hyperparameters

        # get envirionment
        self.env = env
        self.last_state = torch.tensor(self.env.reset(), dtype=torch.double)
        self.__nn()

        # record
        self.record = []

    def __nn(self):
        # define the neural network model for predition
        self.net1 = nn.Sequential(nn.Linear(48 + 1, 256),
                                  nn.Tanh(),
                                  nn.Linear(256, 256),
                                  nn.Tanh(),
                                  nn.Linear(256, 128),
                                  nn.Tanh(),
                                  nn.Linear(128, 64),
                                  nn.Tanh(),
                                  nn.Linear(64, 48 + 1))

        self.net1.apply(self.__init_weights)
        self.loss1 = nn.MSELoss(reduction='sum')
        self.optimizer1 = torch.optim.Adam(self.net1.parameters(), lr=0.001)
        self.net1.to(torch.float64)
        self.net1

    def __init_weights(self, m):
        # initialize the weights of the neural network
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.001)

    def train(self):
        time1 = time.time()
        experience = torch.load('acrobotAgent/data/experience.pt')
        last_state = []
        action = []
        state = []
        reward = []
        for transition in experience:
            last_state.append(transition[0])
            action.append(transition[1])
            state.append(transition[2])
            reward.append(transition[3])

        last_state = torch.stack(last_state)
        action = torch.stack(action).to(torch.float64)
        state = torch.stack(state).to(torch.float64)
        reward = torch.stack(reward)
        time2 = time.time()

        for i in range(1, len(experience)):
            if i % 1000 == 0:
                print(i, '/', len(experience), flush=True)
            # forward pass
            last_state_action = torch.cat([last_state[i], action[i]])
            predicted_state_reward = self.net1(last_state_action)
            # backward pass
            self.optimizer1.zero_grad()
            truth_state_reward = torch.cat([state[i], reward[i]])
            loss = self.loss1(predicted_state_reward, truth_state_reward)
            loss.backward()
            self.optimizer1.step()
            self.record.append(loss.item())
            # print(loss.item())
            pass
        time3 = time.time()
        print(time2 - time1, time3 - time2)

    def generateExperience(self):
        experience = []
        env = self.env
        state = env.reset()
        state = torch.Tensor(state)
        for i in range(1000):
            last_state = state
            action = np.random.choice([1, 2, 3, 4], 1)
            state, reward = env.step(action)

            last_state = torch.tensor(last_state)
            action = torch.tensor(action)
            state = torch.tensor(state)
            reward = torch.tensor(reward)
            experience.append([last_state, action, state, reward])

        torch.save(experience, 'acrobotAgent/data/experience.pt')
        print('experience saved')

    def test(self):
        # test the agent
        env = self.env
        state = env.reset()
        for i in range(100):
            state, _ = env.step([torch.argmax(self.net1(torch.tensor(state))).item() + 1])
            plt.imshow(state.reshape(6, 8))
            plt.show()
            display.display(plt.gcf())
            display.clear_output(wait=True)


config = {
    "n_player": 1,
    "board_width": 8,
    "board_height": 6,
    "n_beans": 5,
    "max_step": 50,
}

thisgame = SnakeGame(config)
state = thisgame.reset()
agent = SnakeAgent(thisgame)
# agent.generateExperience()
agent.train()
plt.plot(agent.record)
plt.show()
