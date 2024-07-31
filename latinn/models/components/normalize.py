import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """
    Module to normalize the inputs prior to training the NN.
    """

    def __init__(self, mean: float = 0, std: float = 1):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        assert std > 0, "std <= 0 for Normalizer"

        super(Normalizer, self).__init__()
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor):
        return x * self.std + self.mean

    def forward(self, x: torch.Tensor):
        return self.normalize(x)
