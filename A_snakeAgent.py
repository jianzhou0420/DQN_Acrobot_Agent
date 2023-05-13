### Provide your code here
import torch
from maai_cwork.examples.common import utils
from comp0124.maai_cwork.env import make
from comp0124.maai_cwork.env import snakes
from maai_cwork.run_utils import get_players_and_action_space_list, run_game
from torch import nn
from comp0124.maai_cwork.env import make
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torchviz

from envZJA.SnakeGame import SnakeGame


class SnakeAgent():
    def __init__(self, env):
        # set hyperparameters
        self.max_episodes = 200
        self.max_actions = 50
        self.gamma = 0.9
        self.exploration_rate = 0.5
        self.exploration_decay = self.exploration_rate / self.max_episodes
        # get envirionment
        self.env = env
        self.last_state = torch.tensor(self.env.reset(), dtype=torch.double)
        self.__nn()

    def __nn(self):
        # define the neural network model for snake 1
        self.net1 = nn.Sequential(nn.Linear(6 * 8, 256),
                                  nn.Linear(256, 256),
                                  nn.Linear(256, 128),
                                  nn.Linear(128, 64),
                                  nn.Linear(64, 4))
        self.net1.apply(self.__init_weights)
        self.loss1 = nn.MSELoss(reduction='sum')
        self.optimizer1 = torch.optim.Adam(self.net1.parameters(), lr=0.001)
        self.net1.to(torch.float64)

    def __init_weights(self, m):
        # initialize the weights of the neural network
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    def train(self):
        # get hyper parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        gamma = self.gamma
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        env = self.env

        for i in range(max_episodes):
            for j in range(max_actions):
                # get the action
                if np.random.rand() < exploration_rate:
                    action = np.random.choice([1, 2, 3, 4], 1)
                else:
                    self.last_state = self.last_state.to(torch.double)
                    action = torch.argmax(self.net1(self.last_state)).item() + 1
                # get the next state
                state, reward = env.step([action])
                state = torch.tensor(state, dtype=torch.double)
                # update the neural network
                self.optimizer1.zero_grad()

                target = reward[0] + gamma * torch.max(self.net1(state))
                loss = self.loss1(self.net1(self.last_state)[action - 1], target)
                loss.backward()
                self.optimizer1.step()
                # update the last state
                self.last_state = state
                # check if the game is over
                if j == 49:
                    print('\ndone No.', i + 1, '/', max_episodes, flush=True)

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
agent.train()
agent.test()
