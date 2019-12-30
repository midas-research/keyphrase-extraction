import torch

from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        tolist = torch.randperm(n).tolist()
        print(tolist)
        return iter(tolist)

    def __len__(self):
        return self.num_samples


# class HackSampler(Sampler):
#     r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
#     If with replacement, then user can specify ``num_samples`` to draw.
#
#     Arguments:
#         data_source (Dataset): dataset to sample from
#     """
#
#     def __init__(self, data_source, block_size=5, plus_window=5):
#         super().__init__(data_source)
#         self.data_source = data_source
#         self.num_samples = len(self.data_source)
#
#         self.block_size = block_size
#         self.plus_window = plus_window
#
#     def __iter__(self):
#         data = [i for i in range(len(self.data_source))]
#         import random
#         blocksize = self.block_size + random.randint(0, self.plus_window)
#
#         # Create blocks
#         blocks = [data[i:i + blocksize] for i in range(0, len(data), blocksize)]
#         # shuffle the blocks
#         random.shuffle(blocks)
#         # concatenate the shuffled blocks
#         data[:] = [b for bs in blocks for b in bs]
#         print(blocksize)
#         print(data[:100])
#         return iter(data)
#
#     def __len__(self):
#         return self.num_samples
