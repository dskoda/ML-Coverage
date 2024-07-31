import itertools
from typing import List, Tuple

import torch
from torch import nn

from .dense_net import make_dense
from .normalize import Normalizer
from .radial import get_radial
from .cutoff import CosineCutoff


class InteractionNet(nn.Module):
    def __init__(
        self,
        input_size: int = 448,
        hidden_layers: int = 3,
        hidden_size: int = 1000,
        hidden_size_decay: float = 1.0,
        n_rbf: int = 6,
        cutoff: float = 6.0,
        radial_fn: str = "gaussian",
        mean: float = 0,
        std: float = 1,
        batch_norm: bool = True,
        cutoff_fn: str = "cosine",
    ):
        super().__init__()

        self.site_energies = make_dense(
            hidden_layers,
            hidden_size,
            input_size,
            batch_norm,
            size_decay=hidden_size_decay,
        )

        self.radial = get_radial(radial_fn, n_rbf=n_rbf, cutoff=cutoff, trainable=True)

        if cutoff_fn is None:
            self.cutoff = nn.Identity()

        elif cutoff_fn == "cosine":
            self.cutoff = CosineCutoff(cutoff)

        interaction_size = input_size * 2 + n_rbf
        self.interaction_energies = make_dense(
            hidden_layers,
            hidden_size,
            interaction_size,
            batch_norm,
            size_decay=hidden_size_decay,
        )

        self.norm = Normalizer(mean, std)

        self.double()

    def forward(self, x, e, d):
        batch_size, n_features = x.size()

        e_site = self.site_energies(x)

        d_emb = self.radial(d.reshape(-1))
        d_emb = d_emb * self.cutoff(d).reshape(-1, 1)
        x_int = torch.cat([x[e[0]], x[e[1]], d_emb], dim=1)

        e_int = self.interaction_energies(x_int)

        return e_site, e_int


class InteractionEnsemble(nn.Module):
    def __init__(
        self,
        *args,
        num_nns: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.models = nn.ModuleList(
            [InteractionNet(*args, **kwargs) for _ in range(num_nns)]
        )

        self.double()

    def forward(self, x, e, d):
        e_site, e_int = [], []

        for m in self.models:
            es, ei = m(x, e, d)
            e_site.append(es)
            e_int.append(ei)

        e_site = torch.stack(e_site).mean(0)
        e_int = torch.stack(e_int).mean(0)

        return e_site, e_int


class PhysInteractionNet(InteractionNet):
    def __init__(
        self,
        input_size: int = 448,
        hidden_layers: int = 3,
        hidden_size: int = 1000,
        hidden_size_decay: float = 1.0,
        n_rbf: int = 6,
        cutoff: float = 6.0,
        radial_fn: str = "gaussian",
        mean: float = 0,
        std: float = 1,
        batch_norm: bool = True,
        exp_coef: Tuple[float, float] = (75.9645, 2.7124),
    ):
        super().__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
            hidden_size_decay=hidden_size_decay,
            n_rbf=n_rbf,
            cutoff=cutoff,
            radial_fn=radial_fn,
            mean=mean,
            std=std,
            batch_norm=batch_norm,
        )

        self.exp_a, self.exp_b = exp_coef

    def phys_repulsion(self, r, min_r=1.7):
        y = self.exp_a * torch.exp(-self.exp_b * r)
        return y * (r > min_r)

    def forward(self, x, e, d):
        batch_size, n_features = x.size()

        e_site = self.site_energies(x)
        e_phys = self.phys_repulsion(d)

        d_emb = self.radial(d.reshape(-1))
        x_int = torch.cat([x[e[0]], x[e[1]], d_emb], dim=1)

        e_int = self.interaction_energies(x_int) + e_phys

        return e_site, e_int


if __name__ == "__main__":
    _ = InteractionNet()
