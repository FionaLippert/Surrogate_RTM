import os
import os.path as osp
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import wandb
from typing import Any, List
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import SubsetRandomSampler, Subset
from operator import mul
from functools import reduce

from rtm import wandb_utils, utils

def join_paths(*paths):
    return osp.join(*[str(p) for p in paths])
def multiply_values(*values):
    return reduce(mul, values)
def add_values(*values):
    return sum(values)

OmegaConf.register_new_resolver("join_paths", join_paths)
OmegaConf.register_new_resolver("add_values", add_values)
OmegaConf.register_new_resolver("multiply_values", multiply_values)
OmegaConf.register_new_resolver("eval", eval)

# main entry point to script
@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    # for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # optional: load transforms from different dataset
    if cfg.dataset.get('transforms_path', None) is not None:
        transforms = utils.load_transforms(cfg.dataset.transforms_path)
    else:
        transforms = None

    # prepare training and test data
    # wandb_utils.prepare_data_and_log(cfg.dataset, cfg.logger.project, transforms)
    wandb_utils.prepare_data_and_save(cfg.dataset, cfg.data_path, transforms)

    # test if loading works
    dataset, train_indices, val_indices = wandb_utils.load_train_dataset(
        cfg.dataset, data_path=cfg.data_path)

if __name__ == "__main__":
    run()