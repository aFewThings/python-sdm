import torch
from torch.utils.data import Dataset

import numpy as np


class EnvironmentalDataset(Dataset):
    def __init__(self, labels, positions, ids, patch_extractor, transform=None):
        self.labels = labels
        self.ids = ids
        self.positions = positions
        self.extractor = patch_extractor
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tensor = self.extractor[self.positions[idx]]
        if self.transform is not None:
            tensor = self.transform(tensor)
        return torch.from_numpy(tensor).float(), self.labels[idx]

    def numpy(self):
        """
        :return: a numpy dataset of 1D vectors
        """
        assert self.extractor.size == 1, 'the patch size should be 1'
        return np.array([torch.flatten(self[i][0]).numpy() for i in range(len(self))]), self.labels

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__ + '(size: {})'.format(len(self))
