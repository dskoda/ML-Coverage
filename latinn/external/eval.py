import torch
import hydra
from typing import List

from torch_geometric.loader import DataLoader
from ase import Atoms
from latinn.models.interaction import InteractionLitModule
from latinn.models.soap import SoapLitModule
from latinn.data.components import SoapDataset


class Evaluator:
    def __init__(self, ckpt_path: str, hparams: dict, swa: bool = True, **kwargs):
        _module = hydra.utils.instantiate(hparams["model"])

        self.model = _module.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **kwargs,
        )
        ckpt = torch.load(ckpt_path)

        if swa and "StochasticWeightAveraging" in ckpt["callbacks"]:
            self.model.load_state_dict(
                ckpt["callbacks"]["StochasticWeightAveraging"]["average_model_state"]
            )

        self.model.eval()
        self.model.freeze()
        self.data_module = hydra.utils.instantiate(hparams["data"])

    def evaluate(self, dset: List[Atoms]):
        processed = self.data_module.process_dataset(dset)
        loader = self.data_module.get_dataloader(processed, shuffle=False)

        pred = []
        true = []
        for batch in loader:
            batch = batch.to(self.model.device)
            out = self.model.inference(batch)
            pred.append(out)
            if batch.y is not None:
                true.append(batch.y.view(-1))

        pred = torch.concat(pred)
        if len(true) > 0:
            true = torch.concat(true)

        return pred, true
