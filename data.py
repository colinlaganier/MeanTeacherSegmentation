import itertools
import os.path
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


class TransformTwice:
    """
    Apply two transforms to the same input
    """
    def __init__(self, transform, noise_transform):
        self.transform = transform
        self.noise_transform = noise_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.noise_transform(inp)
        return out1, out2

class TwoStreamBatchSampler(Sampler):
    """
    Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
    
class SingleStreamBaselineSampler(Sampler):
    """
    Iterate over a single set of values

    An 'epoch' is one iteration through the primary indices.
    This is for baseline computation with a subset of only labeld data.
    """
    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        return (
            primary_batch 
            for primary_batch
            in  grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    """
    Iterate over an iterable once, in a random order
    Args:
        iterable: an iterable
    Returns:
        an iterator
    """
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    """
    Iterate over an iterable in a random order, forever
    Args:
        indices: an iterable
    Returns:
        an iterator
    """
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks
    ex: grouper('ABCDEFG', 3) --> ABC DEF
    Args:
        iterable: an iterable
        n: the size of each chunk
    Returns:
        an iterator
    """
    args = [iter(iterable)] * n
    return zip(*args)
