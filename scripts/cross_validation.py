import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import random_split, SubsetRandomSampler, Subset
from sklearn import model_selection

from rtm import utils

# main entry point
@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    os.makedirs(cfg.result_dir, exist_ok=True)
    print(f'result_dir = {cfg.result_dir}')

    # for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.task.name == 'cross_validation':
        cross_validation(cfg)


def cross_validation(cfg: DictConfig):

    dataset = instantiate(cfg.dataset, train=True)
    device = utils.get_device(**cfg.device)

    kfold = model_selection.KFold(n_splits=cfg.task.n_cv_folds)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

        if cfg.task.fold_id is not None and fold != cfg.task.fold_id:
            # make sure only the specified fold is run
            continue

        print(f'####### Fold {fold+1}/{cfg.task.n_cv_folds} ########')

        # apply normalization based on training set statistics
        dataset.add_normalization(train_ids)
        dataset.add_transforms(*[utils.DoubleToFloat()]*3)
        
        model = instantiate(cfg.model)
        
        def to_raw_targets(outputs, batch=None):
            return dataset.target_transform.inverse(outputs)
            
        model.to_raw_targets = to_raw_targets

        train_loader = instantiate(cfg.dataloader, dataset, 
                                sampler=SubsetRandomSampler(train_ids))
        val_loader = instantiate(cfg.dataloader, Subset(dataset, val_ids), 
                                 batch_size=1, shuffle=False, num_workers=8)

        # fit PCA to training data
        # TODO: do this only if PCA is used
        model.input_dim_reduction.fit(dataset[train_ids]['inputs'].to(device))
        model.output_dim_reduction.fit(dataset[train_ids]['targets'].to(device))
        
        # adjust loss scale based on target variance
        targets = dataset[train_ids]['targets']
        if getattr(model, 'latent_loss', False):
            targets = model.output_dim_reduction.encode(targets.to(device))
        model.loss_scale = targets.std().item()

        print(f'Explained input variance (%): {100 * model.input_dim_reduction.explained_variance_ratio():.4f}')
        print(f'Explained output variance (%): {100 * model.output_dim_reduction.explained_variance_ratio():.4f}')

        # setup trainer
        logger = instantiate(cfg.logger, name=f'cv_fold_{fold+1}')
        trainer = instantiate(cfg.trainer, logger=logger)

        # train model
        trainer.validate(model, val_loader)
        trainer.fit(model, train_loader, val_loader)
    
        # evaluate model on validation set
        trainer.test(model, val_loader)
    
        if isinstance(trainer.logger, WandbLogger):
            # save config to wandb
            # wandb.config.fold = fold
            cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
            wandb.config.update(cfg_resolved)
            wandb.finish()


if __name__ == "__main__":
    run()