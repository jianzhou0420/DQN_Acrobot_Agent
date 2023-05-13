import numpy as np
import matplotlib.pyplot as plt


class SnakeGame():
    """
    Some information about the game:

    1. I/O
    input of the game: [action]
    common return [state, done, reward]

    2. update/step
    update is divided into two parts, doing action and after action. many flags are set for after actions.

    3. some constrains
    supported number of snake : 2
    minimum
    """

    def __init__(self, config):

        # configuration
        self.n_player = config["n_player"]
        self.board_width = config["board_width"]
        self.board_height = config["board_height"]

        self.n_beans = config["n_beans"]
        self.max_step = config["max_step"]

        self.check_config()

        # storage
        self.state = np.zeros(self.board_width * self.board_height, dtype=int)  # stored as a matrix, as it is fixed
        self.__available_state = set(np.arange(self.state.size))  # actually it is an index of the state
        self.snake = [[] for _ in range(self.n_player)]  # stored as a list, as it vary in length in different condition
        self.food = []  # same as above
        self.last_snake = [[] for _ in range(self.n_player)]

        # print
        print("Now please use env.reset() to reset the game")

        # flags
        self.flag_init = False
        self.flag_die = [False for _ in range(self.n_player)]
        self.flag_food = [False for _ in range(self.n_beans)]  # the size of the food is fixed
        self.done = False

        # counter
        self.counter = 0

    def reset(self):
        """
        reset the game

        In specific, two steps
            1. place snakes
            2. place food

        :return: only [state]
        """
        # reset the all storage
        self.state = np.zeros(self.board_height * self.board_width)
        self.__available_state = set(np.arange(self.state.size))
        self.snake = [[] for _ in range(self.n_player)]
        self.food = []
        self.reward = [[] for _ in range(self.n_player)]

        # reset the snake
        for i in range(self.n_player):
            # the initial length of the snake is 3
            self.snake[i] = []
            self.snake[i].append([0, 2 * i])
            self.snake[i].append([0, 2 * i + 1])
            self.snake[i].append([1, 2 * i + 1])  # the initial position of the snake

            index = [
                0 + 2 * i,
                0 + 2 * i + 1,
                1 * self.board_width + 2 * i + 1,
            ]

            for idx in index:
                self.__available_state.remove(idx)
                self.state[idx] = i + 1

        # reset the food
        locations_index = np.random.choice(list(self.__available_state), self.n_beans, replace=False)
        for index in locations_index:
            self.food.append([index // self.board_width, index % self.board_width])
            self.__available_state.remove(index)
            self.state[index] = -1  # food's state is -1

        # reset the done
        self.done = False
        self.counter = 0

        return self.state

    def step(self, actions):
        """
        0. verify the action
        1. update the snake 1.1 update the body 1.2 update the head
        2. flag
        :param action: has the shape of [action1,action2,....]
        :return:
        """
        # TODO: need to finish it
        for i in range(self.n_player):
            self.last_snake[i] = self.snake[i].copy()

        tail = [[] for _ in range(self.n_player)]
        # step0: verify the action's correctness
        for single_action in actions:
            assert 1 <= single_action <= 4, 'the action {} is not in the set of [1,2,3,4]'.format(single_action)

        # step1: update the snake
        # first update the body and then update the head ########

        for i, single_snake in enumerate(self.snake):
            # now single_snake is a list and i+1 is the id of it
            # action means: 1 go left, 2 go up, 3 go right, 4 go down
            # update the body

            # # restore state
            # for single_body in single_snake:
            #     idx_body_in_state = single_body[0] * self.board_width + single_body[1]
            #     self.state[idx_body_in_state] = 0
            #     self.__available_state.add(idx_body_in_state)

            tail[i] = single_snake.pop(-1)  # remove the last one

            # update the head
            head = single_snake[0].copy()
            if actions[i] == 1:  # go left
                head[1] -= 1
            elif actions[i] == 2:  # go up
                head[0] -= 1
            elif actions[i] == 3:  # go right
                head[1] += 1
            elif actions[i] == 4:  # go  down
                head[0] += 1

            boundary_check_head = self.boundaryCheck(row=head[0], col=head[1])
            head = [boundary_check_head // self.board_width, boundary_check_head % self.board_width]
            single_snake.insert(0, head)

            self.snake[i] = single_snake

        # first check what he has eaten or ate nothing
        #  what we need to check to get the reward? 1. if the snake ate its body 2. if collide with each other  3. if it ate any food.
        if self.n_player == 2:  # assume 2 as the maximum of players
            if self.snake[0][0] == self.snake[1][0]:
                self.flag_die = [True, True]

        for i, single_snake in enumerate(self.snake):
            idx_head_in_state = single_snake[0][0] * self.board_width + single_snake[0][1]  # head's idx in state

            if self.state[idx_head_in_state] == 0:  # ate nothing
                self.reward[i] = 0

            elif self.state[idx_head_in_state] == -1:  # ate food
                # reward set to 1, you need to create a new food
                self.reward[i] = 1
                self.snake[i].append(tail[i])  # add the tail back
                # find the idx of the food
                idx_food_in_state = idx_head_in_state
                idx_food_in_map = [idx_head_in_state // self.board_width, idx_head_in_state % self.board_width]
                idx_food_change_in_food_list = self.food.index(idx_food_in_map)
                self.flag_food[idx_food_change_in_food_list] = True  # set the flag of the lost food to be true, means it need to be replaced

            else:  # ate body
                self.flag_die[i] = True
                self.reward[i] = 3 - len(single_snake)

        # handle flag
        self.handleFlag()
        self.flag_die = [False for _ in range(self.n_player)]
        self.flag_food = [False for _ in range(self.n_beans)]
        # construct the new state
        self.state = np.zeros(self.board_height * self.board_width)
        self.__available_state = set(range(self.board_height * self.board_width))
        for i, single_snake in enumerate(self.snake):
            for single_body in single_snake:
                idx_body_in_state = single_body[0] * self.board_width + single_body[1]
                self.state[idx_body_in_state] = i + 1
        for single_food in self.food:
            idx_food_in_state = single_food[0] * self.board_width + single_food[1]
            self.state[idx_food_in_state] = -1
            self.__available_state.remove(idx_food_in_state)

        return self.state, self.reward

        # in the edn handle flag

    def check_config(self):
        """
        TODO: need to finish it, do it in the end
        check the configuration
        :return: None
        """
        assert 0 < self.n_player < 3, "only support 2-player game"
        assert self.board_width >= 8, "board width should be larger than 8"
        assert self.board_height >= 6, "board height should be larger than 6"
        assert self.n_beans >= 5, "number of beans should be larger than 5"
        assert self.max_step >= 50, "max step should be larger than 50"

    def handleFlag(self):
        """

        TODO: finish it
        update the state according to the flag
        :return:
        """
        # what we need to generate a new state?
        # 1. new self.snake
        # 2. new self.food

        # 1. new self.snake
        for i, single_flag_die in enumerate(self.flag_die):
            if single_flag_die == False:
                continue
            # now the snake is dead
            # steps to do: 1.remove the snake from the state 2. respawn it in the state
            # 1. remove the snake from the state
            idx_snake_in_state = []
            for block in self.last_snake[i]:
                idx_snake_in_state.append(block[0] * self.board_width + block[1])
            assert np.all(self.state[idx_snake_in_state] == i + 1), "the snake is not in the state"
            for idx in idx_snake_in_state:
                self.state[idx] = 0

        for i, single_flag_die in enumerate(self.flag_die):
            if single_flag_die == False:
                continue
            # 2. respawn it in the state
            # first find the available state,just find a place to respawn it
            for start_place in range(self.state.size):
                idx1 = self.boundaryCheck(start_place)
                idx2 = self.boundaryCheck(start_place + 1)
                idx3 = self.boundaryCheck(start_place + 1 + self.board_width)
                # print(self.state.reshape(self.board_height, self.board_width))
                if self.state[idx1] <= 0:
                    if self.state[idx2] <= 0:
                        if self.state[idx3] <= 0:

                            # this is the place we spawn the snake
                            # first report food
                            for idx in [idx1, idx2, idx3]:
                                if self.state[idx] == -1:
                                    self.reportFood(idx)

                            # then spawn the snake
                            idx_in_state = [idx1, idx2, idx3]
                            idx_in_matrix = [[idx_in_state[0] // self.board_width, idx_in_state[0] % self.board_width],
                                             [idx_in_state[1] // self.board_width, idx_in_state[1] % self.board_width],
                                             [idx_in_state[2] // self.board_width, idx_in_state[2] % self.board_width]]

                            self.snake[i] = []
                            for single_idx_in_matrix in idx_in_matrix:
                                self.snake[i].append(single_idx_in_matrix)
                            break

        # 2. new self.food
        # find available state
        temp_available = set(np.arange(self.state.size))
        for single_snake in self.snake:
            for single_block in single_snake:
                temp_available.remove(single_block[0] * self.board_width + single_block[1])
        for single_food in self.food:  # no matter it is eaten or not, we can to remove it from the available state,
            try:
                temp_available.remove(single_food[0] * self.board_width + single_food[1])
            except:
                pass  # if eaten, then the block is occupied by the snake, if not, then it is occupied by the food

        num_resapwn_food = np.sum(self.flag_food)
        locations_index = np.random.choice(list(temp_available), num_resapwn_food, replace=False)
        local_counter = 0
        for i in range(len(self.food)):
            if self.flag_food[i] == True:
                this_index = locations_index[local_counter]
                self.food[i] = [this_index // self.board_width, this_index % self.board_width]
                self.__available_state.add(this_index)
                self.state[this_index] = -1  # food's state is -1
                local_counter += 1

    # support functions
    def reportFood(self, idx):
        # have detected a food in this place. 1. report it to flag, set it to snake's body
        idx_food_in_map = [idx // self.board_width, idx % self.board_width]
        idx_food_change_in_food_list = self.food.index(idx_food_in_map)

        self.flag_food[idx_food_change_in_food_list] = True  # set the flag of the lost food to be true, means it need to be replaced

    def spwanSnake(self, i):
        '''
        # TODO: finish it
        :param i:
        :return:
        '''

    def boundaryCheck(self, idx=None, row=None, col=None):
        if idx == None and row == None and col == None:
            raise ValueError("You must specify one of the three parameters")
        if idx != None:
            row = idx // self.board_width
            col = idx % self.board_width
        if row >= 6:
            row = abs(row % 6)
        if col >= 8:
            col = abs(col % 8)

        if col < 0:
            col = 8 + col
        if row < 0:
            row = 6 + row
        new_idx = row * self.board_width + col
        if new_idx >= self.state.size:
            raise ValueError("the new idx is larger than the state size")
        return new_idx
