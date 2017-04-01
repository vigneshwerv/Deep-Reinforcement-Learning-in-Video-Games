class BaseEnvironment(object):
    """Base class for derived environment classes."""

    def __init__(self, **kwargs):
        """ Self-explanatory. """
        raise NotImplementedError("This is a base class! Please use a derived Environment class.")

    def getPreviousObservation(self, **kwargs):
        """ Used by the derived classes to get the previous observation."""
        raise NotImplementedError

    def getReward(self, **kwargs):
        """ Used by the derived classes to obtain the reward from performing the action a
            supplied to performAction.
        """
        raise NotImplementedError

    def getObservation(self, **kwargs):
        """ Used by the derived classes to get the current observation."""
        raise NotImplementedError

    def getActionPerformed(self, **kwargs):
        """ Used by the derived classes to get the most recent action performed."""
        raise NotImplementedError

    def performAction(self, **kwargs):
        """ Used to supply the action to perform in the environment. """
        raise NotImplementedError

    def getPossibleActions(self, **kwargs):
        """ Used to return the list of possible in the current environment. """
        raise NotImplementedError

    def sampleRandomAction(self, **kwargs):
        """ Used to sample random actions from the action space in the current environment. """
        raise NotImplementedError
