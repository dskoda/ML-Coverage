from typing import Any
from typing import List

import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from torchmetrics import MaxMetric
from torchmetrics import MeanMetric
from torchmetrics import MeanSquaredError
from torchmetrics import MinMetric
from torchmetrics import PearsonCorrCoef

from .base import LatINNModule


class SoapLitModule(LatINNModule):
    """LatINN module that uses just the atomic environments."""

    def forward(self, x: torch.Tensor):
        """Computes the contribution energies for each site"""
        e_site = self.net(x)
        return e_site

    def inference(self, batch: Batch):
        e_site = self.forward(batch.x)
        e_site = scatter(e_site.view(-1), batch.batch).view(-1)

        return e_site


if __name__ == "__main__":
    _ = SoapLitModule(None, None, None)
