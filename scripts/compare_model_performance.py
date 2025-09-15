import torch
from rtm import utils
import wandb
import os
import omegaconf
from hydra.utils import instantiate
import timeit
import numpy as np
import pandas as pd

from rtm import torchscript_utils, data_utils

from matplotlib import pyplot as plt
import seaborn as sn

import xarray as xr

# general setup
run = wandb.init()
data_path = '/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/output/datasets'
figure_path = '/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/output/figures'
retrieval_path = '/nfs/Sentinel5/users/fional/OPERA/retrievals/simple_s5_spectrometer_only_1e3_test_set_fixed_LHS'

# Ensure the figure path exists
os.makedirs(figure_path, exist_ok=True)

max_chi2 = 1000

# Define the models to compare
# experiment_name = 'fixed_pca_dropout0001_swir1'
# model_dict = {
#     'MLP-PCA': 
#         {
#             # 'run_path': 'flippert/spectrometer_only_swir1/2o8nvcfb',
#             'run_path': 'flippert/spectrometer_only_swir1/3raslyid',
#             'test_dataset': 'synthetic_remotap_swir1_spectrometer_only_test_set_LHS',
#             # 'retrieval_dirname': 'data_collection_retrieval_mlp-pca-1e6-dropout0001_perturbed_prior'
#             'retrieval_dirname': 'data_collection_retrieval_mlp-pca-1e6-fixed-pca_perturbed_prior'
#         },
#     'Ratio MLP-PCA': 
#         {
#             'run_path': 'flippert/spectrometer_only_swir1/6urvx3d8',
#             'test_dataset': 'synthetic_remotap_swir1_ratio_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_ratio-mlp-pca-1e6-dropout0001_perturbed_prior'
#         },
#     'CNN':
#         {
#             'run_path': 'flippert/spectrometer_only_swir1/9jm0boho',
#             'test_dataset': 'synthetic_remotap_swir1_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_cnn-1e6-dropout0001_perturbed_prior'
#         },
# }

# experiment_name = 'fixed_pca_dropout0001_nir'
# iwin = 1 # nir
# model_dict = {
#     'MLP-PCA': 
#         {
#             'run_path': 'flippert/spectrometer_only_nir/kwtlbzfy',
#             'test_dataset': 'synthetic_remotap_nir_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_mlp-pca-1e6_perturbed_prior'
#         },
#     'Ratio MLP-PCA': 
#         {
#             'run_path': 'flippert/spectrometer_only_nir/jktuef67',
#             'test_dataset': 'synthetic_remotap_nir_ratio_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_ratio-mlp-pca-1e6-dropout0001_perturbed_prior'
#         },
#     'CNN':
#         {
#             'run_path': 'flippert/spectrometer_only_nir/0emw1eq3',
#             'test_dataset': 'synthetic_remotap_nir_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_cnn-1e6-dropout0001_perturbed_prior'
#         },
# }

# experiment_name = 'trainable_pca'
# model_dict = {
#     'MLP-PCA': 
#         {
#             'run_path': 'flippert/spectrometer_only_swir1/dolansci',
#             'test_dataset': 'synthetic_remotap_swir1_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_mlp-pca-1e6_perturbed_prior'
#         },
#     'Ratio MLP-PCA': 
#         {
#             'run_path': 'flippert/spectrometer_only_swir1/ksud9qxf',
#             'test_dataset': 'synthetic_remotap_swir1_ratio_spectrometer_only_test_set_LHS',
#             'retrieval_dirname': 'data_collection_retrieval_ratio_mlp-pca-1e6_perturbed_prior'
#         },
# }

def load_retrieval_results(retrieval_path, retrieval_dirname, npix=1000, iwin=2):
    ds_list = []
    for i in range(npix):
        fp = f'{retrieval_path}/{retrieval_dirname}/pix_00{i+1:>03}_win_{iwin:02}.nc'
        if not os.path.exists(fp):
            continue
        ds_list.append(xr.load_dataset(fp))
    ds_ret_data = xr.concat(ds_list, dim='npix')

    return ds_ret_data

def get_retrieval_residuals_last_iteration(ds_ret_data, normalize='continuum', non_scattering=False):

    if non_scattering:
        pred = ds_ret_data.radiance_ns.values
    else:
        pred = ds_ret_data.radiance_nn.values
    truth = ds_ret_data.radiance_ms.values
    chi2 = ds_ret_data.chi2.values

    valid = (chi2 > 0)
    last_itr = np.sum(valid, axis=1)  - 1

    pred = np.stack([pred[i, :, last_itr[i]] for i in range(pred.shape[0])], axis=0)
    truth = np.stack([truth[i, :, last_itr[i]] for i in range(truth.shape[0])], axis=0)
    chi2 = np.stack([chi2[i, last_itr[i]] for i in range(chi2.shape[0])], axis=0)

    if normalize == 'continuum':
        z = truth.max(axis=1, keepdims=True)
        norm_truth = truth / z
        norm_pred = pred / z
        res = 100 * (norm_pred - norm_truth)
    elif normalize == 'targets':
        res = 100 * (pred - truth) / truth
    else:
        res = pred - truth

    return res


def get_retrieval_residuals_all_iterations(ds_ret_data, normalize='continuum', 
                                           max_chi2=1000, non_scattering=False,
                                           min_data=None, max_data=None):
    if non_scattering:
        pred = ds_ret_data.radiance_ns.values
    else:
        pred = ds_ret_data.radiance_nn.values
    truth = ds_ret_data.radiance_ms.values
    chi2 = ds_ret_data.chi2.values
    inputs = ds_ret_data.nn_x_in.values

    chi2 = ds_ret_data.chi2.values
    mask = np.logical_and(
        chi2 > 0, chi2 < max_chi2
    )

    pred = pred.transpose(0, 2, 1).reshape(-1, pred.shape[1])
    truth = truth.transpose(0, 2, 1).reshape(-1, truth.shape[1])
    inputs = inputs.transpose(0, 2, 1).reshape(-1, inputs.shape[1])
    mask = mask.reshape(-1)

    # only keep data within the specified ranges
    if min_data is not None:
        mask = np.logical_and(mask, np.all(inputs >= min_data, axis=1))
    if max_data is not None:
        mask = np.logical_and(mask, np.all(inputs <= max_data, axis=1))

    pred = pred[mask, :]
    truth = truth[mask, :]
    inputs = inputs[mask, :]

    if normalize == 'continuum':
        z = truth.max(axis=1, keepdims=True)
        norm_truth = truth / z
        norm_pred = pred / z
        res = 100 * (norm_pred - norm_truth)
    elif normalize == 'targets':
        res = 100 * (pred - truth) / truth
    else:
        res = pred - truth

    return res


def load_model_from_wandb(run, run_path):
    """    
    Load a model from a wandb run artifact.

    Args:
        run: wandb run object.
        run_path: path to the model artifact in the format 'name/project/id'.

    Returns:
        forward_model: The loaded forward model.
        checkpoint: The model checkpoint.
    """
    name, project, id = run_path.split('/')
    artifact = run.use_artifact(f'{name}/{project}/model-{id}:best', type='model')
    artifact_dir = artifact.download()
    checkpoint = torch.load(os.path.join(artifact_dir, 'model.ckpt'), map_location=torch.device('cpu'))

    model = utils.load_model_from_checkpoint(checkpoint, map_location=torch.device('cpu'))
    model = model.to('cpu')
    forward_model = model.forward_model
    forward_model.eval()

    return forward_model, checkpoint

def load_test_dataset(checkpoint, data_path):
    """
    Load the test dataset from a wandb run artifact.

    Args:
        checkpoint: model checkpoint to be used for loading the dataset.
        data_path: path to the directory containing all datasets.

    Returns:
        test_dataset: loaded test dataset.
    """
    cfg = checkpoint['hydra_config']
    data_cfg = omegaconf.OmegaConf.create(cfg['dataset'])
    dataset = data_utils.load_test_dataset(data_cfg, data_path)

    return dataset

def make_predictions(forward_model, dataset, non_scattering=False):

    if non_scattering:
        pred = dataset.proxy.detach()
    else:
        pred = []
        for i in range(dataset.pixel_data.shape[0]):
            pixel_data = dataset.pixel_data[i]
            proxy = dataset.proxy[i] if hasattr(dataset, 'proxy') else None
            pred.append(forward_model(pixel_data, proxy).detach())
        pred = torch.stack(pred, dim=0)

    if forward_model.proxy_operation == 'ratio':
        gt = dataset.targets.detach() * dataset.proxy.detach()
    elif forward_model.proxy_operation == 'difference':
        gt = dataset.targets.detach() + dataset.proxy.detach()
    else:
        gt = dataset.targets.detach()

    return pred, gt

def relative_residuals(forward_model, dataset, non_scattering=False, normalize='continuum'):
    pred, gt = make_predictions(forward_model, dataset, non_scattering=non_scattering)

    if normalize == 'continuum':
        z = gt.max(dim=1, keepdim=True).values
        norm_gt = gt / z
        norm_pred = pred / z
        rel_res = 100 * (norm_pred - norm_gt)
    elif normalize == 'targets':
        rel_res = 100 * (pred - gt) / gt
    else:
        rel_res = pred - gt

    return rel_res

def get_data_ranges(dataset):
    """
    Get the input data ranges for the dataset.
    """
    min_data = dataset.pixel_data.min(dim=0).values.numpy()
    max_data = dataset.pixel_data.max(dim=0).values.numpy()

    return min_data, max_data

def plot_histogram(model, dataset, ax, name=None, non_scattering=False, **kwargs):
    rel_res = relative_residuals(model, dataset, non_scattering=non_scattering)
    sn.histplot(rel_res.flatten().detach().numpy(),
                stat='percent', label=name,
                ax=ax, **kwargs)
    
def plot_boxplot(model, dataset, ax, name=None, **kwargs):
    pred, gt = make_predictions(model, dataset)
    rel_res = 100 * (pred - gt) / gt.max(dim=1, keepdim=True).values
    sn.boxplot(data=rel_res.flatten().detach().numpy(), ax=ax, **kwargs)
    ax.set_title(name)
    ax.set_ylabel('Relative residuals [%]')
    ax.grid(True, alpha=0.2)

def plot_residuals(predictions, targets, ax, wl=None, vmin=-10, vmax=10, 
                   normalize='continuum', **ax_kwargs):
    """
    Plot the residuals between predictions and targets as function of wavelength.
    """
    z = targets.max(1, keepdims=True)

    if normalize == 'continuum':
        norm_targets = targets / z
        norm_outputs = predictions / z
        res = 100 * (norm_outputs - norm_targets)
    elif normalize == 'targets':
        res = 100 * (predictions - targets) / targets
    else:
        res = predictions - targets

    mae = np.nanmean(np.abs(res))
    bias = np.nanmean(res)
    std = np.nanstd(res)

    res_median = np.nanmedian(res, axis=0)
    q00 = np.nanquantile(res, 0.00, axis=0)
    q01 = np.nanquantile(res, 0.01, axis=0)
    q05 = np.nanquantile(res, 0.05, axis=0)
    q25 = np.nanquantile(res, 0.25, axis=0)
    q75 = np.nanquantile(res, 0.75, axis=0)
    q95 = np.nanquantile(res, 0.95, axis=0)
    q99 = np.nanquantile(res, 0.99, axis=0)
    q100 = np.nanquantile(res, 1.00, axis=0)

    if wl is None:
        wl = np.arange(res.shape[1])

    l = ax.plot(wl, res_median, label='median')
    ax.fill_between(wl, q25, q75, alpha=0.3, color=l[0].get_color(), label='25-75% quantile range')
    ax.fill_between(wl, q05, q95, alpha=0.15, color=l[0].get_color(), label='5-95% quantile range')
    ax.fill_between(wl, q01, q99, alpha=0.05, color=l[0].get_color(), label='1-99% quantile range')
    # ax.fill_between(wl, q00, q100, alpha=0.1, color='lightgray', label='0-100% quantile range')
    
    ax.annotate(f'MAE = {mae:.2f}%', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
    ax.annotate(f'Bias = {bias:.2f}%', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)
    ax.annotate(f'Std = {std:.2f}%', xy=(0.05, 0.7), xycoords='axes fraction', fontsize=10)

    ax.grid(True, alpha=0.2)
    ax.set_ylim(vmin, vmax)
    ax.set(**ax_kwargs)

if __name__ == "__main__":
    # Load models and datasets
    models = {}
    datasets = {}
    for name, config in model_dict.items():
        forward_model, checkpoint = load_model_from_wandb(run, config['run_path'])
        dataset = load_test_dataset(checkpoint, data_path)
        models[name] = forward_model
        datasets[name] = dataset

    # Plot histograms of relative residuals
    bins = np.linspace(-10, 10, 200)
    ncols = len(models) #+ 1
    fig, ax = plt.subplots(ncols=ncols, figsize=(5*ncols, 3), sharey=True)
    fig_box, ax_box = plt.subplots(figsize=(3*ncols, 3))

    df = {'dataset': [], 'residuals': [], 'model': []}


    for i, (name, model) in enumerate(models.items()):
        ax[i].set_title(name)
        rel_res = relative_residuals(model, datasets[name]).numpy()
        min_data, max_data = get_data_ranges(datasets[name])
        label = f'test set (bias = {np.nanmean(rel_res.flatten()):.2f}%, std = {np.nanstd(rel_res.flatten()):.2f}%)'
        sn.histplot(rel_res.flatten(),
                    stat='percent', label=label,
                    ax=ax[i], bins=bins, alpha=1.0, fill=False, element='step', lw=2)
        
        # df = {'dataset': ['test set']*len(rel_res.flatten()), 
        #       'residuals': rel_res.flatten()}
        df['model'].extend([name]*len(rel_res.flatten()))
        df['dataset'].extend(['test set']*len(rel_res.flatten()))
        df['residuals'] = np.concatenate((df['residuals'], rel_res.flatten()))

        ret_results = load_retrieval_results(retrieval_path, model_dict[name]['retrieval_dirname'], iwin=iwin)
        rel_res = get_retrieval_residuals_all_iterations(ret_results, 
                                                         normalize='continuum', 
                                                         max_chi2=max_chi2,
                                                         min_data=min_data, 
                                                         max_data=max_data)
        # rel_res = get_retrieval_residuals_last_iteration(ret_results, normalize='continuum')
        label = f'retrieval (bias = {np.nanmean(rel_res.flatten()):.2f}%, std = {np.nanstd(rel_res.flatten()):.2f}%)'
        sn.histplot(rel_res.flatten(),
                stat='percent', label=label,
                ax=ax[i], bins=bins, alpha=1.0, fill=False, element='step', lw=2)
        
        df['model'].extend([name]*len(rel_res.flatten()))
        df['dataset'].extend(['retrieval']*len(rel_res.flatten()))
        df['residuals'] = np.concatenate((df['residuals'], rel_res.flatten()))

        # rel_res = get_retrieval_residuals_last_iteration(ret_results, normalize='continuum')
        # sn.histplot(rel_res.flatten(),
        #         stat='percent', label='retrieval (last iteration)',
        #         ax=ax[i], bins=bins, alpha=0.3, fill=False, element='step', lw=2)

        ax[i].set_xlabel('radiance error [%]')
        ax[i].set_xlim(-1.5, 1.5)
        # ax[i].set_ylim(0, 15)

        ax[i].axvline(0, color='k', linestyle='--', linewidth=1)
        ax[i].legend(loc='upper left')

    # # last model is non-scattering approximation
    # ax[-1].set_title('Non-scattering approximation')
    # rel_res = relative_residuals(models[name], datasets[name], non_scattering=True).numpy()
    # label = f'test set (bias = {np.nanmean(rel_res.flatten()):.2f}%, std = {np.nanstd(rel_res.flatten()):.2f}%)'
    # sn.histplot(rel_res.flatten(),
    #             stat='percent', label=label,
    #             ax=ax[-1], bins=bins, alpha=1.0, fill=False, element='step', lw=2)
    
    # # df = {'dataset': ['test set']*len(rel_res.flatten()), 
    # #       'residuals': rel_res.flatten()}

    # df['model'].extend(['non-scattering']*len(rel_res.flatten()))
    # df['dataset'].extend(['test set']*len(rel_res.flatten()))
    # df['residuals'] = np.concatenate((df['residuals'], rel_res.flatten()))
    
    # # plot_histogram(models[name], datasets[name], ax[-1], name='test set', non_scattering=True, alpha=0.3, bins=bins, fill=False, element='step', lw=2)
    # rel_res = get_retrieval_residuals_all_iterations(ret_results, 
    #                                                  normalize='continuum', 
    #                                                  non_scattering=True, 
    #                                                  max_chi2=max_chi2, 
    #                                                  min_data=min_data, 
    #                                                  max_data=max_data)
    # rel_res = get_retrieval_residuals_last_iteration(ret_results, normalize='continuum', non_scattering=True)
    # label = f'retrieval (bias = {np.nanmean(rel_res.flatten()):.2f}%, std = {np.nanstd(rel_res.flatten()):.2f}%)'
    # sn.histplot(rel_res.flatten(),
    #             stat='percent', label=label,
    #             ax=ax[-1], bins=bins, alpha=1.0, fill=False, element='step', lw=2)

    # rel_res = get_retrieval_residuals_last_iteration(ret_results, normalize='continuum', non_scattering=True)
    # sn.histplot(rel_res.flatten(),
    #             stat='percent', label='retrieval (last iteration)',
    #             ax=ax[-1], bins=bins, alpha=0.3, fill=False, element='step', lw=2)

    # df['model'].extend(['non-scattering']*len(rel_res.flatten()))
    # df['dataset'].extend(['retrieval']*len(rel_res.flatten()))
    # df['residuals'] = np.concatenate((df['residuals'], rel_res.flatten()))

    # boxplot for test set
    df = pd.DataFrame(df)
    sn.boxplot(x='model', y='residuals', data=df, hue='dataset', 
               ax=ax_box, showfliers=False, 
               width=0.5, gap=0.1, linewidth=1.5)
    # ax_box.set_title('test set')
    ax_box.set_ylabel('radiance error [%]')
    ax_box.grid(True, axis='y', alpha=0.2)

    # sn.boxplot(x='model', y='residuals', data=df.query('dataset == "retrieval"'), 
    #            ax=ax_box[1], showfliers=False, width=0.5, linewidth=1.5)
    # ax_box[1].set_title('retrieval')
    # ax_box[1].set_ylabel('Relative residuals [%]')
    # ax_box[1].grid(True, axis='y', alpha=0.2)

    ax[-1].set_xlabel('radiance error [%]')
    ax[-1].set_xlim(-1.5, 1.5)
    # ax[-1].set_ylim(0, 15)
    ax[-1].axvline(0, color='k', linestyle='--', linewidth=1)
    ax[-1].legend(loc='upper left')

    # ax[0].legend(loc='upper left')
    
    fig.savefig(os.path.join(figure_path, f'relative_residual_distributions_{experiment_name}_max_chi2={max_chi2}.png'), 
                dpi=300, bbox_inches='tight')
    
    fig_box.savefig(os.path.join(figure_path, f'relative_residual_boxplots_per_dataset_{experiment_name}_max_chi2={max_chi2}.png'), 
                   dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    plt.close(fig_box)





    # ax.set_xlabel('Relative residuals [%]')
    # ax.set_xlim(-1, 1)
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_path, 'relative_residual_distribution.png'), dpi=300)
    # plt.close()

    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # for name, model in models.items():
    #     ret_results = load_retrieval_results(retrieval_path, model_dict[name]['retrieval_dirname'], iwin=iwin)
    #     rel_res = get_retrieval_residuals_all_iterations(ret_results, normalize='continuum')
    #     sn.histplot(rel_res.flatten(),
    #             stat='percent', label=name,
    #             ax=ax, bins=bins, alpha=0.3)
    # ax.set_xlabel('Relative residuals [%]')
    # ax.set_xlim(-1, 1)
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_path, 'relative_residual_distribution_retrieval_all_itr.png'), dpi=300)
    # plt.close()

    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # for name, model in models.items():
    #     ret_results = load_retrieval_results(retrieval_path, model_dict[name]['retrieval_dirname'], iwin=iwin)
    #     rel_res = get_retrieval_residuals_last_iteration(ret_results, normalize='continuum')
    #     sn.histplot(rel_res.flatten(),
    #             stat='percent', label=name,
    #             ax=ax, bins=bins, alpha=0.3)
    # ax.set_xlabel('Relative residuals [%]')
    # ax.set_xlim(-1, 1)
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_path, 'relative_residual_distribution_retrieval_last_itr.png'), dpi=300)
    # plt.close()


