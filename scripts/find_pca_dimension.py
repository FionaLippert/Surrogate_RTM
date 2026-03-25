from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import os
import os.path as osp
from functools import reduce
import omegaconf
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from operator import mul
from functools import reduce
from rtm import data_utils

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

def transform_targets(data, log_mean=None, log_std=None):
    log_data = np.log(data)
    if log_mean is None:
        log_mean = log_data.mean(axis=0, keepdims=True)
    if log_std is None:
        log_std = log_data.std(axis=0, keepdims=True)
    return (log_data - log_mean) / log_std, log_mean, log_std

def inverse_transform_targets(data, log_mean, log_std):
    log_data = data * log_std + log_mean
    return np.exp(log_data)

def reconstruction_error(pca, data, log_mean, log_std, z='targets'):
    # data transformation
    norm_data, _, _ = transform_targets(data, log_mean, log_std)

    # apply PCA
    outputs = pca.inverse_transform(pca.transform(norm_data))
    outputs = inverse_transform_targets(outputs, log_mean, log_std)

    if z == 'continuum':
        rel_res = 100 * (outputs - data) / data.max(axis=1, keepdims=True)
    else:
        rel_res = 100 * (outputs - data) / data

    rel_rmse = np.sqrt(np.mean((rel_res)**2))

    return rel_rmse

def reconstruction_bias(pca, data, log_mean, log_std, z='targets'):
    # data transformation
    norm_data, _, _ = transform_targets(data, log_mean, log_std)

    # apply PCA
    outputs = pca.inverse_transform(pca.transform(norm_data))
    outputs = inverse_transform_targets(outputs, log_mean, log_std)

    if z == 'continuum':
        rel_res = 100 * (outputs - data) / data.max(axis=1, keepdims=True)
    else:
        rel_res = 100 * (outputs - data) / data

    rel_bias = np.mean(rel_res)

    return rel_bias

# main entry point
@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def run(cfg: DictConfig):

    figure_path = os.path.join(cfg.result_path, 'pca_reconstruction_errors')
    os.makedirs(figure_path, exist_ok=True)

    # load training/validation data
    dataset, train_indices, val_indices = data_utils.load_train_dataset(
        cfg.dataset, data_path=cfg.data_path)
    
    # get subset of training/validation data
    n_subset = int(cfg.task.data_frac * len(train_indices))
    train_indices = np.random.choice(train_indices, n_subset, replace=False)
    train_targets = dataset.targets.numpy()[train_indices]
    val_targets = dataset.targets.numpy()[val_indices]

    # fit PCA and compute reconstruction error for different numbers of components
    errors = []
    n_max = train_targets.shape[1]
    n_range = np.arange(0, n_max, cfg.task.step_size)
    n_range[0] = 1 # we need at least one component
    df = {'band': [], 'n_components': [], 'rel_rmse': [], 'rel_bias': [], 'explained_var': []}
    for n in n_range:
        pca = PCA(n_components=n)

        # transform training data and fit PCA
        norm_data, log_mean, log_std = transform_targets(train_targets)
        pca.fit(norm_data)

        # get explained variance
        explained_var = np.sum(pca.explained_variance_ratio_) * 100

        # evaluate reconstruction error on validation set
        err = reconstruction_error(pca, val_targets, log_mean, log_std, z=cfg.task.normalization)
        bias = reconstruction_bias(pca, val_targets, log_mean, log_std, z=cfg.task.normalization)
        print(f'  n_components={n}, rel_rmse={err:.8f}%, explained_var={explained_var:.8f}%')
        df['band'].append(cfg.dataset.band)
        df['n_components'].append(n)
        df['rel_rmse'].append(err)
        df['rel_bias'].append(bias)
        df['explained_var'].append(explained_var)

        errors.append(reconstruction_error(pca, val_targets, log_mean, log_std))

    df = pd.DataFrame(df)
    info = f'data_frac={cfg.task.data_frac}_normalization={cfg.task.normalization}'
    fname = f'{cfg.dataset.name}_pca_reconstruction_error_{info}.csv'
    df.to_csv(os.path.join(figure_path, fname), index=False)

    print(f'Saved PCA reconstruction error data to {os.path.join(figure_path, fname)}')


if __name__ == '__main__':
    run()