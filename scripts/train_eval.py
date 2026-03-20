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

from rtm import data_utils, torchscript_utils, utils, dimensionality_reduction, layers

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
OmegaConf.register_new_resolver("len", len)

# run with CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# main entry point
@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def run(cfg: DictConfig):

    # for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    print('GPU available?', torch.cuda.is_available())
    print("CPU cores reserved by Slurm:", os.environ.get("SLURM_CPUS_PER_TASK"))

    # for debugging
    if cfg.task.get('detect_anomalies', False):
        torch.autograd.set_detect_anomaly(True) # detect nans
        torch.set_flush_denormal(True) # detect tiny numbers

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
    model = instantiate(cfg.model)
    trained = False

    # optional: load pretrained model weights
    if cfg.task.get('model_path', None) is not None:
        checkpoint = torch.load(cfg.task.model_path)
        print('Loading model state from checkpoint...')
        model.load_model_state(checkpoint)

        # remember that this is a pretrained model
        trained = True

    if 'train' in cfg.task.name:
        train(trainer, model, cfg, pretrained=trained)
        trained = True
    else:
        trained = False


    if 'eval' in cfg.task.name:
        if not trained:
            # skip evaluation if the model has not been trained or loaded with pretrained weights
            print('Model has not been trained or loaded with pretrained weights. Skipping evaluation.')
        else:
            eval(trainer, model, cfg)

    
    if isinstance(trainer.logger, WandbLogger):
        # save config to wandb
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

        # add number of trainable parameters to wandb config
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg_resolved['num_parameters'] = num_params
        print(f'num parameters = {num_params}')

        wandb.config.update(cfg_resolved)
        wandb.finish()


def train(trainer: pl.Trainer, model: pl.LightningModule, cfg: DictConfig, pretrained: bool):

    if cfg.verbose:
        print('####### Start model training ########')
    
    # get training data
    dataset, train_indices, val_indices = data_utils.load_train_dataset(
        cfg.dataset, data_path=cfg.data_path)
    
    # possibly select smaller subset of the training/validation data
    if cfg.task.get('data_frac', 1.0) < 1.0:
        # subsample training data
        n_subset = int(cfg.task.data_frac * len(train_indices))
        train_indices = np.random.choice(train_indices, n_subset, replace=False)
        # subsample validation data
        n_subset = int(cfg.task.data_frac * len(val_indices))
        val_indices = np.random.choice(val_indices, n_subset, replace=False)

    # get transforms
    if not pretrained:
        transforms = data_utils.prepare_transforms(cfg.transforms, 
                                                dataset, 
                                                train_indices, 
                                                **cfg.dataset.normalization)
        model.set_transforms(transforms)
    else:
        transforms = model.transforms

        # make sure to overwrite jacobian weights
        model.jacobian_weights = torch.tensor(cfg.dataset.jacobian_weights, dtype=torch.float32)


    # update config with final transforms (including normalization and double-to-float conversion)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg_resolved['transforms'] = transforms.__dict_repr__()

    # save this for later reference
    model.save_hydra_config(cfg_resolved)

    # setup data loaders
    train_loader = instantiate(cfg.dataloader, dataset, 
                               sampler=SubsetRandomSampler(train_indices))
    val_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                             batch_size=1, shuffle=False)

    print(f'Using {len(train_indices)} datapoints for training')
    print(f'Using {len(val_indices)} datapoints for validation')

    train_data = dataset[train_indices]
    device = utils.get_device(**cfg.device)
    
    if not pretrained:
        # fit PCA to training data, if applicable
        if isinstance(getattr(model.forward_model, 'encoder', None), layers.PCAEncoder):
            print('Fitting input PCA...')
            inputs = model.forward_model.input_transform(train_data['inputs'])
            model.forward_model.encoder.fit(inputs)
            exp_var = 100 * model.forward_model.encoder.explained_variance_ratio()
            print(f'Explained input variance (%): {exp_var:.4f}')

        if isinstance(getattr(model.forward_model, 'proxy_module', None), layers.PCAEncoder):
            print('Fitting proxy PCA...')
            proxy = model.forward_model.proxy_transform(train_data['proxy_targets'])
            model.forward_model.proxy_module.fit(proxy)
            exp_var = 100 * model.forward_model.proxy_module.explained_variance_ratio()
            print(f'Explained proxy variance (%): {exp_var:.4f}')

        if isinstance(getattr(model.forward_model, 'profile_module', None), layers.PCAEncoder):
            print('Fitting profile PCA...')
            profiles = model.forward_model.profile_transform(train_data['profile_inputs'])
            model.forward_model.profile_module.fit(profiles)
            exp_var = 100 * model.forward_model.profile_module.explained_variance_ratio()
            print(f'Explained profile variance (%): {exp_var:.4f}')

        if isinstance(getattr(model.forward_model, 'decoder', None), layers.PCADecoder):
            print('Fitting output PCA...')
            targets = model.forward_model.target_transform(train_data['targets'])
            model.forward_model.decoder.fit(train_data['targets'])
            exp_var = 100 * model.forward_model.decoder.explained_variance_ratio()
            print(f'Explained output variance (%): {exp_var:.4f}')

    
        # adjust loss scale based on target variance
        targets = train_data['targets']
        if model.forward_model.target_transform is not None:
            targets = model.forward_model.target_transform(targets)

    # train model
    trainer.validate(model, val_loader)
    trainer.fit(model, train_loader, val_loader)

    # trace model and save as TorchScript (for CPU usage in Fortran)
    # first, prepare dummy inputs used for tracing
    input_vars = ['inputs']
    if cfg.task.trace_with_proxy_input:
        input_vars.append('proxy_targets')
    if cfg.task.trace_with_profile_input:
        input_vars.append('profile_inputs')
    dummy_input = {k: v for k, v in dataset[0].items() if k in input_vars}

    # if applicable, trace the model with TorchScript and save it
    if cfg.task.get('trace_model', False):
        fp = osp.join(cfg.output_path, f'{cfg.model.name}_{cfg.dataset.name}.pt')
        torchscript_utils.trace_model_and_save(
            model.forward_model,
            dummy_input,
            filepath=fp,
            device_type=('cuda' if cfg.task.trace_with_gpu else 'cpu'),
            )
        # also save corresponding wavelengths for reference
        fp = osp.join(cfg.output_path, f'{cfg.model.name}_{cfg.dataset.name}_wavelengths.dat')
        np.savetxt(fp, dataset.wavelengths, delimiter=' ')

        if cfg.verbose:
            print(f'Traced model saved to {fp}')


def eval(trainer: pl.Trainer, model: pl.LightningModule, cfg: DictConfig):
    
    if cfg.verbose:
        print('####### Start model evaluation ########')

    # use CPU for testing if required
    if cfg.task.get('measure_cpu_time', False):
        logger = instantiate(cfg.logger)
        test_trainer = instantiate(cfg.trainer, 
                            accelerator='cpu',
                            devices=1,
                            logger=logger, 
                            default_root_dir=cfg.output_path)
        
        print(f'num threads before changing them: {torch.get_num_threads()}')
        
        # explicitly set number of threads to know what runtime is actually measured
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        torch.set_num_threads(num_cpus)
        torch.set_num_interop_threads(1)
        num_workers = 1
        if cfg.verbose:
            print(f'Measuring model runtime on {num_cpus} CPUs')
    else:
        num_workers = cfg.dataloader.num_workers
        logger = instantiate(cfg.logger)
        test_trainer = instantiate(cfg.trainer, 
                            logger=logger, 
                            devices=1, # to make sure all predictions are returned
                            default_root_dir=cfg.output_path,
                            )

    if cfg.task.final_eval:
        # get test data
        dataset = data_utils.load_test_dataset(cfg.dataset, 
                                                data_path=cfg.data_path, 
                                                # transforms=transforms,
                                                # run=wandb.run
                                                )

        # setup data loader
        test_loader = instantiate(cfg.dataloader, dataset, batch_size=1, 
                                  shuffle=False, num_workers=num_workers)
        predict_loader = instantiate(cfg.dataloader, dataset, batch_size=1, 
                                  shuffle=False)

        print(f'Using {len(dataset)} data points for final testing')
    else:
        # get validation data
        dataset, _, val_indices = data_utils.load_train_dataset(
            cfg.dataset, data_path=cfg.data_path)

        # possibly select smaller subset of the validation data
        if cfg.task.get('data_frac', 1.0) < 1.0:
            # subsample validation data
            n_subset = int(cfg.task.get('data_frac', 1.0) * len(val_indices))
            val_indices = np.random.choice(val_indices, n_subset, replace=False)
            print(len(val_indices))

        # setup data loader
        test_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                                  batch_size=1, shuffle=False, num_workers=num_workers)
        predict_loader = instantiate(cfg.dataloader, Subset(dataset, val_indices), 
                                  batch_size=1, shuffle=False)

        print(f'Using {len(val_indices)} data points for validation/prediction')

    # evaluate model
    test_trainer.test(model, test_loader)

    # also save individual predictions
    predictions = test_trainer.predict(model, predict_loader)
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