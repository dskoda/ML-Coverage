from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import (
    MaxMetric,
    MinMetric,
    MeanMetric,
    MeanSquaredError,
    PearsonCorrCoef,
)
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from .base import LatINNModule


class InteractionLitModule(LatINNModule):
    """LatINN module that uses interactions between adsorbates."""

    def forward(self, xi: torch.Tensor, eij: torch.Tensor, dij: torch.Tensor):
        """Computes the contribution energies for each site"""
        e_site, e_int = self.net(xi, eij, dij)
        return e_site, e_int

    def inference(self, batch: Batch):
        e_site, e_int = self.forward(batch.x, batch.edge_index, batch.edge_attr)
        e_site = scatter(e_site.view(-1), batch.batch).view(-1)
        e_out = torch.zeros(batch.batch.max() + 1, device=e_int.device, dtype=e_int.dtype)
        e_int = scatter(e_int.view(-1), batch.batch[batch.edge_index[0]], out=e_out).view(-1)

        energies = e_site + e_out

        return energies


if __name__ == "__main__":
    _ = InteractionLitModule(None, None, None)
