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

from rtm import utils, dimensionality_reduction

def join_paths(*paths: List[str]) -> str:
    return osp.join(*[str(p) for p in paths])

OmegaConf.register_new_resolver("join_paths", join_paths)

# main entry point
@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    # for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # setup logger
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.logger.save_dir, exist_ok=True)
    logger = instantiate(cfg.logger)
    cfg.output_path = logger.experiment.dir

    print(f'Results will be saved to {cfg.output_path}')

    # setup trainer
    trainer = instantiate(cfg.trainer, 
                          logger=logger, 
                          default_root_dir=cfg.output_path)

    for cb in trainer.callbacks:
        # set checkpoint dir to wandb dir
        if isinstance(cb, pl.callbacks.ModelCheckpoint):
            cb.dirpath = osp.join(cfg.output_path, 'checkpoints')
            break

    # setup model
    print(cfg.model)
    model = instantiate(cfg.model)

    # optional: load pretrained model weights
    if cfg.model.get('model_path', None) is not None:
        path_to_model = wandb.use_model(cfg.model.model_path)
        model.load_state_dict(torch.load(path_to_model))

    trained = False

    if 'train' in cfg.task.name:
        transforms = train(trainer, model, cfg)
        trained = True
    if 'eval' in cfg.task.name:
        eval(trainer, model, cfg, transforms, trained=trained)
    
    if isinstance(trainer.logger, WandbLogger):
        # save config to wandb
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update(cfg_resolved)
        wandb.finish()


def train(trainer: pl.Trainer, model: pl.LightningModule, cfg: DictConfig):

    if cfg.verbose:
        print('####### Start model training ########')

    # get training data
    dataset = instantiate(cfg.dataset, train=True)

    # split training data into training and validation sets
    train_indices, val_indices = dataset.get_data_split(cfg.dataset.val_size)

    # apply normalization based on training set statistics
    dataset.add_normalization(train_indices)
    dataset.add_transforms(*[utils.DoubleToFloat()]*3)
    
    # define function to convert model outputs to physical targets
    # def to_raw_targets(outputs, batch=None):
    #     return dataset.target_transform.inverse(outputs)
    model.target_transform = dataset.target_transform

    # setup data loaders
    train_loader = instantiate(cfg.dataloader, dataset, 
                               sampler=SubsetRandomSampler(train_indices))
    val_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                             batch_size=1, shuffle=False)

    # fit PCA to training data, if applicable
    train_data = dataset[range(len(dataset))]
    device = utils.get_device(**cfg.device)
    if isinstance(getattr(model, 'input_dim_reduction', None), dimensionality_reduction.PCA):
        model.input_dim_reduction.fit(train_data['inputs'].to(device))
        exp_var = 100 * model.input_dim_reduction.explained_variance_ratio()
        print(f'Explained input variance (%): {exp_var:.4f}')

    if isinstance(getattr(model, 'output_dim_reduction', None), dimensionality_reduction.PCA):
        model.output_dim_reduction.fit(train_data['targets'].to(device))
        exp_var = 100 * model.output_dim_reduction.explained_variance_ratio()
        print(f'Explained output variance (%): {exp_var:.4f}')

    
    # adjust loss scale based on target variance
    targets = train_data['targets']
    if getattr(model, 'latent_loss', False) and model.output_dim_reduction is not None:
        targets = model.output_dim_reduction.encode(targets.to(device))
    model.loss_scale = targets.std().item()

    # train model
    trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)

    return dataset.get_transforms()


def eval(trainer: pl.Trainer, model: pl.LightningModule, cfg: DictConfig,
         transforms: List[utils.RTMTransform] = [None, None, None], trained: bool = False):
    
    if cfg.verbose:
        print('####### Start model evaluation ########')

    if cfg.task.final_eval:
        # get test data
        dataset = instantiate(cfg.dataset, train=False)

        # make sure transforms (incl. normalization) are the same as for training
        dataset.set_transforms(*transforms)

        # setup data loader
        test_loader = instantiate(cfg.dataloader, dataset, batch_size=1, shuffle=False)
    else:
        # get validation data
        dataset = instantiate(cfg.dataset, train=True)
        _, val_indices = dataset.get_data_split(cfg.dataset.val_size)
        
        # make sure transforms (incl. normalization) are the same as for training
        dataset.set_transforms(*transforms)

        # setup data loader
        test_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                                  batch_size=1, shuffle=False)


    # evaluate model
    kwargs = {'ckpt_path': 'best'} if cfg.task.load_best_model and trained else {}
    trainer.test(model, test_loader, **kwargs)

    # also save individual predictions
    predictions = trainer.predict(model, test_loader, **kwargs)
    outputs, targets = map(list, zip(*predictions))
    outputs = np.concatenate(outputs, 0)
    targets = np.concatenate(targets, 0)

    # calculate evaluation metrics
    eval_metrics = {}
    eval_metrics['wavelength'] = dataset.wavelengths
    eval_metrics['MAE'] = utils.mae(outputs, targets, axis=0)
    eval_metrics['RMSE'] = utils.rmse(outputs, targets, axis=0)
    eval_metrics['NRMSE'] = utils.nrmse(outputs, targets, z=np.mean(targets, axis=0), axis=0)
    eval_metrics['MAPE'] = utils.mape(outputs, targets, axis=0)
    eval_metrics['SMAPE'] = utils.smape(outputs, targets, axis=0)
    eval_metrics['rel_bias'] = utils.relative_bias(outputs, targets, axis=0)
    eval_metrics['MAPE_continuum'] = utils.mape_continuum(outputs, targets, axis=0)
    eval_metrics['RMSE_continuum'] = utils.rmse_continuum(outputs, targets, axis=0)
    eval_metrics['rel_bias_continuum'] = utils.relative_bias(outputs, targets, z=targets.max(1, keepdims=True), axis=0)
    eval_metrics['R2'] = utils.r2_score(outputs, targets, axis=0)

    # metrics on log-transformed data
    # ensure that outputs are positive
    outputs = np.maximum(outputs, np.finfo(outputs.dtype).eps)
    eval_metrics['R2_log'] = utils.r2_score(np.log(outputs), np.log(targets), axis=0)
    

    # save to csv
    eval_metrics = pd.DataFrame(eval_metrics)
    fname = f'eval_metrics_{wandb.run.id}.csv'
    eval_metrics.to_csv(osp.join(cfg.output_path, fname), index=False)

    if cfg.task.save_predictions:
        # save individual predictions
        fname = f'predictions_{wandb.run.id}.npy'
        np.save(osp.join(cfg.output_path, fname), outputs)

        # save ground truth
        fname = f'targets_{wandb.run.id}.npy'
        np.save(osp.join(cfg.output_path, fname), targets)



if __name__ == "__main__":
    run()