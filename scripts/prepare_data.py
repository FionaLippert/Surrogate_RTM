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

from rtm import wandb_utils

def join_paths(*paths: List[str]) -> str:
    return osp.join(*[str(p) for p in paths])

OmegaConf.register_new_resolver("join_paths", join_paths)

# main entry point to script
@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    # for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # prepare training and test data
    wandb_utils.prepare_data_and_log(cfg.dataset, cfg.logger.project)


if __name__ == "__main__":
    run()