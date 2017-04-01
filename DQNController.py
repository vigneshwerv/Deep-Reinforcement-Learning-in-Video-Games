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
        return

    def __sample_epsilon_action__(self, action):
        if random.random() < self.epsilon:
            sampled_action = self.environment.sampleRandomAction()
            while sampled_action == action:
                sampled_action = self.environment.sampleRandomAction()
            action = sampled_action
        # Epsilon annealing
        self.__anneal_epsilon__()
        return action

    def __supply_action_to_environment__(self, action, action_q_value):
        # Get action using epsilon-greedy strategy
        action = self.__sample_epsilon_action__(action)
        for i in xrange(self.repeat_action):
            self.environment.performAction(action)
        # Add current state, action, reward, consequent state to experience replay
        next_state = self.current_state[1:]
        next_state.append(self.environment.getObservation())
        # TODO: Write this line when experience replay is done
        # ExperienceReplay.add(current_state,
        #                       action,
        #                       self.environment.getReward(),
        #                       next_state,
        #                       self.environment.isTerminalState())
        return

    def __compute_loss__(self, q_values, target_q_values):
        # Compute a y_j (Refer to Algorithm 1) tensor to be supplied to the training_network
        for i in xrange(len(self.minibatch_terminal)):
            if self.minibatch_terminal[i]:
                # If it is the terminal episode, just use the reward
                loss_tensor.append(self.minibatch_reward[i])
            else:
                # Otherwise, use reward + discount * max_q_value
                loss_tensor.append(self.minibatch_reward[i]
                                   + (self.discount * max(target_q_values[i])))
        loss_tensor = (loss_tensor - self.q_values) ** 2
        # TODO: Write code for this when the training network is ready
        # Call network's gradient descent step

    def run(self):
        """This method will be called from the main() method."""
        # Initialize initial state by sampling actions randomly and performing
        # them on the environment
        while len(self.current_state) != self.stack_num:
            random_action = self.environment.sampleRandomAction()
            self.environment.performAction(random_action)
            self.current_state.append(self.environment.getObservation())

        t = 0
        C = 0
        while True:
            # Use the current state of the emulator and predict an action which gets
            # added to replay memory (use playing_network)
            q_values = self.playing_network.predict(current_state)
            action_to_perform = np.argmax(q_values, axis=1)
            # Perform action based on epsilon-greedy search and store the transitions
            # in experience replay
            self.__supply_action_to_environment__(action_to_perform, q_values[action_to_perform])
            # If in the observation phase, do not train the network just yet
            t += 1
            if t < self.observation_time_steps:
                continue
            # Sample minibatch of size self.minibatch_size from experience replay
            # TODO: Write this line when experience replay is done
            # This minibatch should have states, actions, rewards, nextStates,
            # and whether nextState was final or not
            q_values = self.training_network.predict(minibatch_states)
            target_q_values = self.playing_network.predict(minibatch_next_states)
            self.__compute_loss__(q_values, target_q_values)
            if C % self.copy_steps == 0:
                # Copy weights from training network to playing network
                # TODO: Code for this needs to be written; handled by network
            C += 1
