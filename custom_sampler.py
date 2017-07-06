from torch.utils.data.sampler import Sampler

class NAT_sampler(Sampler):
    """Sequentially samples elements from a randomly generated list of indices, without replacement.

        Arguments:
            indices (list): a list of indices
        """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def update(self, new_indices):
        self.indices = new_indices