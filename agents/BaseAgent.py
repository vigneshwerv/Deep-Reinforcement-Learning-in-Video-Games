class BaseAgent(object):
    """Base class for derived agent classe."""

    def __init__(self, **kwargs):
        """ Self-explanatory. """
        raise NotImplementedError("This is a base class! Please use a derived Agent class.")

    def predict(self, **kwargs):
        """ Used by the derived classes to predict an action."""
        raise NotImplementedError

    def compute_loss(self, **kwargs):
        """ Used by the derived classes to compute loss on the network."""
        raise NotImplementedError

    def save_network(self, **kwargs):
        """ Used by the derived classes to save the network. This should be called
            by the Agent Controller.
        """
        raise NotImplementedError
