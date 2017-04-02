import numpy as np

class ExperienceReplay(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.memories = None

    def add(self, memory):
        if self.memories is None:
            self.memories = np.array([memory])
            return
        if (len(self.memories) + 1) > self.maxlen:
            self.memories = self.memories[1:]
        self.memories = np.append(self.memories, [memory], axis=0)
        return

    def sample(self, num_samples):
        return self.memories[
                np.random.randint(
                    self.memories.shape[0],
                    size=num_samples)]
