import numpy as np
import random

class ExperienceReplay(object):

    def __init__(self, maxlen, dimensions, minibatch_size, stack_num):
        self.maxlen = maxlen
        self.dimensions = dimensions
        self.minibatch_size = minibatch_size
        self.stack_num = stack_num

        self.screens = np.empty((maxlen,) + self.dimensions, dtype=np.uint8)
        self.actions = np.empty(maxlen, dtype=np.uint8)
        self.rewards = np.empty(maxlen, dtype=np.integer)
        self.terminals = np.empty(maxlen, dtype=np.bool)
        self.count = 0
        self.current = 0

        self.prestates = np.empty((self.minibatch_size, self.stack_num) + self.dimensions, dtype=np.uint8)
        self.poststates = np.empty((self.minibatch_size, self.stack_num) + self.dimensions, dtype=np.uint8)

    def add(self, memory):
        screen, action, reward, terminal = memory
        self.screens[self.current, ...] = screen
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.maxlen
        return

    def getStackedState(self, index):
        index = index % self.count

        if index >= self.stack_num - 1:
            return self.screens[(index - (self.stack_num - 1)):(index + 1), ...]
        else:
            indices = [(index - 1) % self.count for i in reversed(range(self.stack_num))]
            return self.screens[indices, ...]

    def sample(self):
        indices = []
        while len(indices) < self.minibatch_size:
            while True:
                index = random.randint(self.stack_num, self.maxlen - 1)
                if index >= self.current and (index - self.stack_num) < self.current:
                    continue
                if self.terminals[(index - self.stack_num):index].any():
                    continue
                break
            self.prestates[len(indices), ...] = self.getStackedState(index - 1)
            self.poststates[len(indices), ...] = self.getStackedState(index)
            indices.append(index)

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return self.prestates, actions, rewards, self.poststates, terminals
