import json
import os
import sys
import cv2
import copy
import pandas as pd
import numpy as np

import torch.utils.data.sampler

class BalancedSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, img_paths, batch_size, shuffle, idxs):
        self.img_paths = img_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idxs = idxs

        self._setup()

    def _setup(self):
        labels_list = [path.split('/')[-1].split('\\')[0] for path in np.array(self.img_paths)[self.idxs]]

        scenting_indices = [l_i for l_i, label in enumerate(labels_list) if label == 'scenting']
        non_scenting_indices = [l_i for l_i, label in enumerate(labels_list) if label == 'non_scenting']

        self.lookup_src = {
            "scenting"     : scenting_indices,
            "non_scenting" : non_scenting_indices
        }

        self.num_examples = len(self.idxs)

    def init_lookup(self):
        lookup = copy.deepcopy(self.lookup_src)
        if self.shuffle:
            for key, val in lookup.items():
                np.random.shuffle(val)
        return lookup

    def sample(self, lookup, key):
        key_idxs = lookup[key]
        return np.random.choice(key_idxs)

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        lookup = self.init_lookup()
        for i in range(self.num_examples):
            if i % 2 == 0:
                sample = self.sample(lookup, 'scenting')
            else:
                sample = self.sample(lookup, 'non_scenting')
            yield sample

#============================================================================================

class SubsetIdentitySampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (idx for idx in self.indices)

    def __len__(self):
        return len(self.indices)
