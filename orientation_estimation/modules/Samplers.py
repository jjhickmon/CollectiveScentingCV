import torch
import numpy as np
import copy

class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, classifications, batch_size, shuffle):
        self.classifications = classifications
        self.num_examples = len(classifications)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._setup()

    def _setup(self):
        self.lookup_src = {
            "scenting"     : np.where(self.classifications=='scenting')[0],
            "non_scenting" : np.where(self.classifications=='non_scenting')[0]
        }

    def init_lookup(self):
        lookup = copy.deepcopy(self.lookup_src)

        if self.shuffle:
            for key, val in lookup.items():
                np.random.shuffle(val)

        return lookup

    def sample(self, lookup, key):
        key_idxs = lookup[key]
        return np.random.choice(key_idxs)

    def __iter__(self):
        lookup = self.init_lookup()
        for i in range(self.num_examples):
            if i % 2 == 0:
                sample = self.sample(lookup, 'scenting')
            else:
                sample = self.sample(lookup, 'non_scenting')
            yield sample
