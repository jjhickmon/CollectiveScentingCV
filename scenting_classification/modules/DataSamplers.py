import json
import os
import sys
import cv2
import copy
import pandas as pd
import numpy as np

import torch.utils.data.sampler

class BalancedSubsetSampler_2(torch.utils.data.Sampler):
    ''' For sampling images cropped and already stored in memory. '''
    def __init__(self, img_paths, labels_list, idxs):
        self.img_paths = img_paths
        self.idxs = idxs
        self.labels_list = labels_list
        
        self._setup()
        
    def _setup(self):
        self.idx_labels_list = np.array(self.labels_list)[self.idxs]
        self.scenting_indices = [self.idxs[l_i] for l_i, label in enumerate(self.idx_labels_list) if label == 'scenting']
        self.non_scenting_indices = [self.idxs[l_i] for l_i, label in enumerate(self.idx_labels_list) if label == 'non_scenting']
        self.num_examples = len(self.idx_labels_list)
   
    def __len__(self):
        return self.num_examples
    
    def __iter__(self):
        for i in range(self.num_examples):
            if i % 2 == 0:
                sample = np.random.choice(self.scenting_indices)
            else:
                sample = np.random.choice(self.non_scenting_indices)
            yield sample

#============================================================================================

class BalancedSubsetSampler(torch.utils.data.Sampler):
    ''' For processing imgs in dataset on the fly '''
    def __init__(self, df, batch_size, shuffle, idxs):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idxs = idxs
        self.subset_df = df.iloc[idxs]
        self.classification_series = self.subset_df['classification']

        self._setup()

    def _setup(self):
        local_scenting_idxs = np.where(self.classification_series=='scenting')[0]
        global_scenting_idxs = list(self.subset_df.iloc[local_scenting_idxs].index)

        local_non_scenting_idxs = np.where(self.classification_series=='non_scenting')[0]
        global_non_scenting_idxs = list(self.subset_df.iloc[local_non_scenting_idxs].index)

        self.lookup_src = {
            "scenting"     : global_scenting_idxs,
            "non_scenting" : global_non_scenting_idxs
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
