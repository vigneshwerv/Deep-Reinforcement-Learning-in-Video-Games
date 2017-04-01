from environments import AtariEnvironment.AtariEnvironment as AtariEnvironment

import numpy as np
import random

class DQNController(object):

    def __init__(self, **kwargs):
        self.copy_steps = 10
        self.stack_num = 4
        self.repeat_action = 4
        self.minibatch_size = 32
        self.min_epsilon = 0.1
        self.epsilon = 1.0
        self.anneal_till = 100000
        self.environment = AtariEnvironment()
        self.observation_time_steps = 500
        self.training_network = DQNAgent()
        self.playing_network = DQNAgent()
        self.current_state = [self.environment.getObservation()]

    def __anneal_epsilon__():
        self.epsilon = max(((self.epsilon - self.min_epsilon) / self.anneal_till),
                           self.min_epsilon)

    def __supply_action_to_environment__(self, action):

        if random.random() < self.epsilon:
            sampled_action = self.environment.sampleRandomAction()
            while sampled_action == action:
                sampled_action = self.environment.sampleRandomAction()
            action = sampled_action
            
        self.__anneal_epsilon__()
        for i in xrange(self.repeat_action):
            self.environment.performAction(action)
            # Add current state, action, reward, consequent state to experience replay
            # next_state = self.current_state[1:]
            # next_state.append(self.environment.getObservation())
            # ExperienceReplay.add(current_state,
            #                       action,
            #                       self.environment.getReward(),
            #                       next_state)

    def __compute_loss__(self):
        # Compute a y_j (Refer to Algorithm 1) tensor to be supplied to the training_network

    def run(self):
        """This method will be called from the main() method."""
        # Initialize initial state by sampling actions randomly and performing
        # them on the environment
        while len(self.current_state) != self.stack_num:
            random_action = self.environment.sampleRandomAction()
            self.environment.performAction(random_action)
            self.current_state.append(self.environment.getObservation())

        t = 0
        while t < self.observation_time_steps:
            softmax_outputs = self.playing_network.predict(current_state)
            action_to_perform = np.argmax(softmax_outputs)
            self.__supply_action_to_environment__(action_to_perform)
            t += 1

        C = 0
        while True:
            if C % self.copy_steps == 0:
                # Copy weights from training network to playing network
            # Sample minibatch of size self.minibatch_size from experience replay
            softmax_outputs = self.playing_network.predict(minibatch_states)
            actions_to_perform = np.argmax(softmax_outputs)
            for i in xrange(self.minibatch_size):
                self.__supply_action_to_environment__(actions_to_perform[i])
            self.__compute_loss__()
            C += 1
