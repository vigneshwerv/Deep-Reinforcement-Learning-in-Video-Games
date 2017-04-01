from BaseEnvironment import BaseEnvironment
from PIL import Image

import gym
import numpy as np

class AtariEnvironment(BaseEnvironment):
    """Wrapper class for the OpenAI Gym Atari Environment."""

    def __init__(self, render=True, **kwargs):
        self.environment = gym.make('Pong-v0')
        self.width = 84
        self.height = 84
        self.possible_actions = self.environment.action_space
        if (type(self.possible_actions) != type(gym.spaces.discrete.Discrete(0))):
            raise AssertionError("The sample action space does not consist of Discrete actions.")
        self.previous_observation = None
        self.recent_observation = self.__preprocess_observation__(self.environment.reset())
        self.should_render = render
        if self.should_render:
            self.environment.render()

    def getPreviousObservation(self, **kwargs):
        if self.previous_observation is None:
            raise AssertionError("self.previous_observation is None.")
        return self.previous_observation

    def getReward(self, **kwargs):
        return self.recent_reward

    def getObservation(self, **kwargs):
        return self.recent_observation

    def getActionPerformed(self, **kwargs):
        return self.recent_action

    def performAction(self, **kwargs):
        self.recent_action = kwargs.get('action')
        observation, reward, done, _ = self.environment.step(self.recent_action)
        self.previous_observation = self.recent_observation
        self.recent_observation = self.__preprocess_observation__(observation)
        self.recent_reward = reward
        self.is_done = done
        if self.should_render:
            self.environment.render()
        return True

    def getPossibleActions(self, **kwargs):
        return self.possible_actions.n

    def sampleRandomAction(self, **kwargs):
        return self.possible_actions.sample()

    def __preprocess_observation__(self, observation):
        """This method is to preprocess observation images.

        The RGB images from OpenAI are converted CMYK images and the luminance (Y)
        channel is extracted, downsampled to a width and height of 84x84 using
        Anti-aliasing, and returned.
        """
        rgb_img = Image.fromarray(observation)
        luminance_channel = rgb_img.convert('CMYK').split()[2].resize(
                                (self.width, self.height),
                                Image.ANTIALIAS)
        return np.array(luminance_channel)
