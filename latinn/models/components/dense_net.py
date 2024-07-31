import itertools
from typing import List

import torch
from torch import nn

from .normalize import Normalizer


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def make_dense(
    hidden_layers: int,
    hidden_size: int,
    input_size: int,
    batch_norm: bool = True,
    output_size: int = 1,
    size_decay: float = 1.0,
):
    """Creates a DenseNet from input configs"""
    _layers = [
        int(round(hidden_size * (size_decay ** layer), 0))
        for layer in range(hidden_layers)
    ]
    _layers = [input_size] + _layers + [output_size]

    nn_layers = []
    for i, (inp, out) in enumerate(pairwise(_layers), 1):
        nn_layers += [nn.Linear(inp, out)]
        if i < len(_layers) - 1:
            if batch_norm:
                nn_layers += [nn.BatchNorm1d(out)]

            nn_layers += [nn.Mish()]

    return nn.Sequential(*nn_layers)


class DenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 448,
        hidden_layers: int = 3,
        hidden_size: int = 1000,
        mean: float = 0,
        std: float = 1,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.net = make_dense(
            hidden_layers, hidden_size, input_size, batch_norm
        )
        self.norm = Normalizer(mean, std)

        self.double()

    def forward(self, x):
        y = self.net(x)
        y = self.norm.denormalize(y)
        return y


if __name__ == "__main__":
    _ = DenseNet()
