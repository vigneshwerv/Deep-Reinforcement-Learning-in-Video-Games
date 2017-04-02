from agents.DQNAgent import DQNAgent
from collections import deque
from environments.AtariEnvironment import AtariEnvironment
from ExperienceReplay import ExperienceReplay

import numpy as np
import random

class DQNController(object):

    def __init__(self, **kwargs):
        # Number of steps of training before training network's weights are
        # copied to target network (C)
        self.copy_steps = 10000
        # Number of frames to be stacked for a state representation (m)
        self.stack_num = 4
        # Number of times actions are to be repeated (k)
        self.repeat_action = 4
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
        self.observation_time_steps = 500
        # The network
        self.network = DQNAgent(self.action_space)
        # The current state of the environment (stacked)
        self.current_state = deque(maxlen=self.stack_num)
        self.current_state.append(self.environment.getObservation())
        # Experience replay
        self.memory_limit = 50000
        self.experience_replay = ExperienceReplay(self.memory_limit)
        # Maximum no-ops
        self.num_no_op = 0
        self.max_no_op = 30

    def __anneal_epsilon__(self):
        self.epsilon = max(self.epsilon - ((self.epsilon - self.min_epsilon) / self.anneal_till),
                           self.min_epsilon)
        return

    def __sample_epsilon_action__(self, action):
        if random.random() < self.epsilon:
            sampled_action = None
            while True:
                sampled_action = self.environment.sampleRandomAction()
                if (sampled_action == 0) or (sampled_action == 1):
                    if self.num_no_op == self.max_no_op:
                        continue
                    self.num_no_op += 1
                if sampled_action == action:
                    continue
                break
            action = sampled_action
        # Epsilon annealing
        self.__anneal_epsilon__()
        return action

    def __supply_action_to_environment__(self, action):
        # Get action using epsilon-greedy strategy
        action = self.__sample_epsilon_action__(action)
        for i in xrange(self.repeat_action):
            self.environment.performAction(action=action)
        # Add current state, action, reward, consequent state to experience replay
        next_state = deque(self.current_state, maxlen=self.stack_num)
        next_state.append(self.environment.getObservation())
        self.experience_replay.add((np.stack(self.current_state, axis=2),
                              action,
                              self.environment.getReward(),
                              np.stack(next_state, axis=2),
                              self.environment.isTerminalState()))
        self.current_state = deque(next_state, maxlen=self.stack_num)
        return

    def __compute_loss__(self, minibatch_states, target_q_values, minibatch_action, minibatch_reward, minibatch_terminals):
        # Compute a y_j (Refer to Algorithm 1) tensor to be supplied to the training_network
        loss_tensor = []
        actions = []
        for i in xrange(len(minibatch_terminals)):
            temp_actions = [0] * self.action_space
            temp_actions[minibatch_action[i]] = 1
            actions.append(temp_actions)
            if minibatch_terminals[i]:
                # If it is the terminal episode, just use the reward
                loss_tensor.append(minibatch_reward[i])
            else:
                # Otherwise, use reward + discount * max_q_value
                loss_tensor.append(minibatch_reward[i]
                                   + (self.discount * max(target_q_values[i])))
        # print np.array(actions).shape
        # Call network's gradient descent step
        return self.network.train(minibatch_states, loss_tensor, np.array(actions, dtype=np.float32))

    def run(self):
        """This method will be called from the main() method."""
        # Initialize initial state by sampling actions randomly and performing
        # them on the environment
        while len(self.current_state) != self.stack_num:
            random_action = self.environment.sampleRandomAction()
            self.environment.performAction(action=random_action)
            self.current_state.append(self.environment.getObservation())

        t = 0
        while True:
            # Use the current state of the emulator and predict an action which gets
            # added to replay memory (use playing_network)
            q_values = self.network.predict([np.stack(self.current_state, axis=2)])
            action_to_perform = np.argmax(q_values, axis=1)[0]
            # Perform action based on epsilon-greedy search and store the transitions
            # in experience replay
            self.__supply_action_to_environment__(action_to_perform)
            # If in the observation phase, do not train the network just yet
            t += 1
            if t < self.observation_time_steps:
                continue
            # print "Started training"
            # Sample minibatch of size self.minibatch_size from experience replay
            minibatch = self.experience_replay.sample(self.minibatch_size)
            minibatch_states, minibatch_action, minibatch_reward, minibatch_next_states, \
                    minibatch_terminals = \
                    minibatch[:,0] , minibatch[:,1], minibatch[:,2], minibatch[:,3], minibatch[:,4]
            # print "States minibatch: ", minibatch_states[0][0].shape
            minibatch_states = np.stack([minibatch[i,0] for i in xrange(self.minibatch_size)], axis=0)
            # Fetch Q-values for the next state from the playing network
            minibatch_next_states = np.stack([minibatch[i,3] for i in xrange(self.minibatch_size)], axis=0)
            target_q_values = self.network.predict(minibatch_next_states)
            # COmpute loss
            cost = self.__compute_loss__(
                    minibatch_states, target_q_values, minibatch_action, minibatch_reward, minibatch_terminals)
            if t % 100 == 0:
                print "Cost at iteration", t, " is", cost
                print "Value of epsilon is", self.epsilon
            if t % self.copy_steps == 0:
                print "Copying weights from training network to playing network"
                # Copy weights from training network to playing network
                self.network.copy_weights()
