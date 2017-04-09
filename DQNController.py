from agents.DQNAgent import DQNAgent
from collections import deque
from environments.AtariEnvironment import AtariEnvironment
from ExperienceReplay import ExperienceReplay

import logging
import numpy as np
import random
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DQNController(object):

    def __init__(self, **kwargs):
        # Number of steps of training before training network's weights are
        # copied to target network (C)
        self.copy_steps = 10000
        # Number of frames to be stacked for a state representation (m)
        self.stack_num = 4
        # Number of times actions are to be repeated (k)
        self.repeat_action = 1
        # Size of minibatch
        self.minibatch_size = 32
        # Lower than this, epsilon is kept constant
        self.min_epsilon = 0.1
        # Epsilon's starting value
        self.epsilon = 1.0
        # Number of steps to anneal epsilon
        self.anneal_till = 100000
        # Discount factor
        self.discount = 0.99
        # Variable that holds the current Environment
        self.environment = AtariEnvironment()
        self.action_space = self.environment.getPossibleActions()
        # For how long should the network observe before playing?
        self.observation_time_steps = 50000
        # The network
        self.network = DQNAgent(self.action_space, self.discount)
        self.train_frequency = 4
        self.record_frequency = 10000
        # The current state of the environment (stacked)
        self.current_state = deque(maxlen=self.stack_num)
        self.current_state.append(self.environment.getObservation())
        # Experience replay
        self.memory_limit = 50000
        self.experience_replay = ExperienceReplay(self.memory_limit,
                                                  (84, 84),
                                                  self.minibatch_size,
                                                  self.stack_num)
        # Maximum no-ops
        self.num_no_op = 0
        self.max_no_op = 30
        self.steps = 0

        self.num_epochs = 30
        self.train_steps_per_epoch = 250000

    def __anneal_epsilon__(self):
        if self.steps < self.anneal_till:
            self.epsilon - self.steps * (self.epsilon - self.min_epsilon) / self.anneal_till
        else:
            self.min_epsilon
        return

    def __sample_epsilon_action__(self):
        if random.random() < self.epsilon:
            sampled_action = self.environment.sampleRandomAction()
            action = sampled_action
        else:
            # Use the current state of the emulator and predict an action which gets
            # added to replay memory (use playing_network)
            q_values = self.network.predict(self.experience_replay.getCurrentState())
            action = np.argmax(q_values, axis=1)[0]
        return action

    def __supply_action_to_environment__(self, action):
        self.environment.performAction(action)
        # Add current state, action, reward, consequent state to experience replay
        self.experience_replay.add((self.environment.getObservation(),
                                    action,
                                    self.environment.getReward(),
                                    self.environment.isTerminalState()))
        return

    def __observe__(self):
        observe_start = time.time()
        for _ in xrange(self.observation_time_steps):
            action = self.environment.sampleRandomAction()
            self.__supply_action_to_environment__(action)
        observe_duration = time.time() - observe_start
        logger.info('Finished observation. Steps=%d; Time taken=%.2f',
                self.observation_time_steps, observe_duration)

    def run(self):
        """This method will be called from the main() method."""
        # Observe the game by randomly sampling actions from the environment
        # and performing those actions
        self.__observe__()
        for i in xrange(self.num_epochs):
            self.environment.resetStatistics()
            time_now = time.time()
            for j in xrange(self.train_steps_per_epoch):
                # Get action using epsilon-greedy strategy
                action = self.__sample_epsilon_action__()
                # Perform action based on epsilon-greedy search and store the transitions
                # in experience replay
                self.__supply_action_to_environment__(action)
                if j % self.train_frequency == 0:
                    # print "Started training"
                    # Sample minibatch of size self.minibatch_size from experience replay
                    minibatch = self.experience_replay.sample()
                    minibatch_states, minibatch_action, minibatch_reward, minibatch_next_states, \
                            minibatch_terminals = minibatch
                    cost = self.network.train_network(minibatch_states,
                                                      minibatch_action,
                                                      minibatch_reward,
                                                      minibatch_terminals,
                                                      minibatch_next_states)
                if j % self.record_frequency == 0:
                    self.network.record_average_qvalue(
                            self.experience_replay.getCurrentState(),
                            i * self.train_steps_per_epoch + j)
                # Epsilon annealing
                self.__anneal_epsilon__()
                # if self.time_step % 1000 == 0:
                #     print "Cost at iteration", self.time_step, " is", cost
                #     print "Value of epsilon is", self.epsilon
                self.steps += 1
                if j % self.copy_steps == 0:
                    self.network.copy_weights(i, j)
            total_score, num_games = self.environment.getStatistics()
            time_taken = (time.time() - time_now)
            logger.info("Finished epoch %d: Steps=%d; Time taken=%.2f",
                    i, j, time_taken)
            logger.info("Number of games: %d; Average reward: %.2f", num_games, (total_score / num_games))
            logger.info("Final epsilon value for epoch: %f", self.epsilon)
