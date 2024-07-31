from omegaconf import DictConfig


def fix_DictConfig(cfg: DictConfig):
    """fix all vars in the cfg config
    this is an in-place operation"""
    keys = list(cfg.keys())
    for k in keys:
        if type(cfg[k]) is DictConfig:
            fix_DictConfig(cfg[k])
        else:
            setattr(cfg, k, getattr(cfg, k))
