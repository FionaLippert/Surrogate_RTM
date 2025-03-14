import os
import os.path as osp
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import wandb
from typing import Any, List, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import SubsetRandomSampler, Subset
from operator import mul
from functools import reduce

from rtm import data_utils, utils, dimensionality_reduction, layers

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

# run with CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


##################################################################
## Notes on computing gradients w.r.t. inputs for retrievals
#
# inputs.requires_grad = True
# optimizer.zero_grad()
# output = model(data)
# output.backward()
# grad = inputs.grad
##################################################################



# main entry point
@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    # for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    print('GPU available?', torch.cuda.is_available())

    # setup logging
    os.makedirs(cfg.result_path, exist_ok=True)
    os.makedirs(cfg.data_path, exist_ok=True)
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
    if cfg.task.get('model_path', None) is not None:
        # path_to_model = wandb.use_model(cfg.task.model_path)
        # model.load_state_dict(torch.load(path_to_model))
        checkpoint = torch.load(cfg.task.model_path)
        print('Loading model state from checkpoint...')
        model.load_model_state(checkpoint)

    if 'train' in cfg.task.name:
        # transforms = train(trainer, model, cfg)
        train(trainer, model, cfg)
        trained = True
    else:
        trained = False
        # transforms = None

    # # optional: load transforms explicitly
    # if cfg.task.get('normalization_path', None) is not None:
    #     print(f'Loading transforms from {cfg.task.normalization_path}...')
    #     transforms = utils.load_transforms(cfg.task.normalization_path)

    if 'eval' in cfg.task.name:
        # eval(trainer, model, cfg, transforms, trained=trained)
        eval(trainer, model, cfg, trained=trained)
    
    if isinstance(trainer.logger, WandbLogger):
        # save config to wandb
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update(cfg_resolved)
        wandb.finish()


def train(trainer: pl.Trainer, model: pl.LightningModule, cfg: DictConfig):

    if cfg.verbose:
        print('####### Start model training ########')
    
    # get training data
    dataset, train_indices, val_indices = data_utils.load_train_dataset(
        cfg.dataset, data_path=cfg.data_path, run=wandb.run)
    
    # possibly select smaller subset of the training data
    if cfg.task.get('train_frac', 1.0) < 1.0:
        n_subset = int(cfg.task.train_frac * len(train_indices))
        train_indices = np.random.choice(train_indices, n_subset, replace=False)

    # get transforms
    transforms = data_utils.prepare_transforms(cfg.transforms, 
                                               dataset, 
                                               train_indices, 
                                               **cfg.dataset.normalization)
    model.transforms = transforms

    # set function to convert model outputs to physical targets
    # TODO: adjust if pixel and profile inputs are handled separately!
    # model.input_transform = dataset.pixel_transform
    # model.target_transform = dataset.target_transform
    # model.proxy_transform = dataset.proxy_transform

    # setup data loaders
    train_loader = instantiate(cfg.dataloader, dataset, 
                               sampler=SubsetRandomSampler(train_indices))
    val_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                             batch_size=1, shuffle=False)

    # fit PCA to training data, if applicable
    train_data = dataset[train_indices]
    device = utils.get_device(**cfg.device)
    if isinstance(getattr(model, 'input_dim_reduction', None), 
                  (dimensionality_reduction.PCA, layers.PCAEncoder)):
        model.input_dim_reduction.fit(train_data['inputs'].to(device))
        exp_var = 100 * model.input_dim_reduction.explained_variance_ratio()
        print(f'Explained input variance (%): {exp_var:.4f}')

    if isinstance(getattr(model.forward_model, 'encoder', None), layers.PCAEncoder):
        inputs = model.pixel_transform(train_data['inputs'])
        model.forward_model.encoder.fit(inputs)
        exp_var = 100 * model.forward_model.encoder.explained_variance_ratio()
        print(f'Explained input variance (%): {exp_var:.4f}')

    if isinstance(getattr(model, 'output_dim_reduction', None), 
                  (dimensionality_reduction.PCA, layers.PCADecoder)):
        model.output_dim_reduction.fit(train_data['targets'].to(device))
        exp_var = 100 * model.output_dim_reduction.explained_variance_ratio()
        print(f'Explained output variance (%): {exp_var:.4f}')

    if isinstance(getattr(model.forward_model, 'decoder', None), layers.PCADecoder):
        targets = model.target_transform(train_data['targets'])
        model.forward_model.decoder.fit(train_data['targets'])
        exp_var = 100 * model.forward_model.decoder.explained_variance_ratio()
        print(f'Explained output variance (%): {exp_var:.4f}')

    
    # adjust loss scale based on target variance
    targets = train_data['targets']
    if model.target_transform is not None:
        targets = model.target_transform(targets)
    # if getattr(model, 'latent_loss', False) and model.output_dim_reduction is not None:
    #     targets = model.output_dim_reduction.encode(targets.to(device))
    model.loss_scale = targets.std().item()

    # train model
    trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)

    # return dataset.get_transforms()


def eval(trainer: pl.Trainer, model: pl.LightningModule, cfg: DictConfig,
        #  transforms: Optional[List[utils.RTMTransform]] = None, 
         trained: bool = False):
    
    if cfg.verbose:
        print('####### Start model evaluation ########')

    if cfg.task.final_eval:
        # get test data
        dataset = data_utils.load_test_dataset(cfg.dataset, 
                                                data_path=cfg.data_path, 
                                                # transforms=transforms,
                                                run=wandb.run)

        # setup data loader
        test_loader = instantiate(cfg.dataloader, dataset, batch_size=1, shuffle=False)
    else:
        # get validation data
        dataset, train_indices, val_indices = data_utils.load_train_dataset(
            cfg.dataset, data_path=cfg.data_path, run=wandb.run)

        # setup data loader
        test_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                                  batch_size=1, shuffle=False)

    # # set transform if untrained model
    # if not trained:
    #     model.target_transform = dataset.target_transform
    #     model.proxy_transform = dataset.proxy_transform

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
    fp = osp.join(cfg.output_path, 'eval_metrics.csv')
    eval_metrics.to_csv(fp, index=False)

    # save as wandb artifact
    evaluation = wandb.Artifact(
            f'evaluation-{wandb.run.id}', type="evaluation",
            description="Performance metrics per wavelength, evaluated on independent test set")
    evaluation.add_file(fp)
    wandb.run.log_artifact(evaluation)
    


    if cfg.task.save_predictions:
        # save individual predictions
        fp_outputs = osp.join(cfg.output_path, f'predictions.npy')
        np.save(fp_outputs, outputs)

        # save ground truth
        fp_targets = osp.join(cfg.output_path, f'targets.npy')
        np.save(fp_targets, targets)

        # save corresponding wavelengths
        fp_wavelengths = osp.join(cfg.output_path, 'wavelengths.npy')
        np.save(fp_wavelengths, dataset.wavelengths)

        # save as wandb artifact
        prediction = wandb.Artifact(
            f'prediction-{wandb.run.id}', type="prediction",
            description="Predicted spectra and corresponding targets")
        prediction.add_file(fp_outputs)
        prediction.add_file(fp_targets)
        prediction.add_file(fp_wavelengths)
        wandb.run.log_artifact(prediction)




if __name__ == "__main__":
    run()