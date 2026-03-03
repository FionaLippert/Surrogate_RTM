from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import os
import omegaconf
import pandas as pd
from rtm import data_utils

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
    # pca.fit(norm_data)
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
    # pca.fit(norm_data)
    outputs = pca.inverse_transform(pca.transform(norm_data))
    outputs = inverse_transform_targets(outputs, log_mean, log_std)

    if z == 'continuum':
        rel_res = 100 * (outputs - data) / data.max(axis=1, keepdims=True)
    else:
        rel_res = 100 * (outputs - data) / data

    rel_bias = np.mean(rel_res)

    return rel_bias


if __name__ == '__main__':

    # normalization = 'targets'
    normalization = 'continuum'
    # dataset_type = f'ratio_spectrometer_only_z={normalization}_train_val'
    # dataset_type = f'grasp_spectrometer_only_z={normalization}_train_val'
    # dataset_type = f'ratio_spectrometer_only_matched_retrieval_era5_z={normalization}_train_val'
    # dataset_type = f'spectrometer_only_matched_retrieval_era5_paper_z={normalization}_train_val'
    dataset_type = f'ratio_spectrometer_only_matched_retrieval_era5_paper_z={normalization}_train_val'
    train_frac = 1.0

    # data_dict = {
    #     r'NIR ($N_{\lambda}=211$)': 'synthetic_remotap_nir_ratio_spectrometer_only_LHS',
    #     r'SWIR1 ($N_{\lambda}=851$)': 'synthetic_remotap_swir1_ratio_spectrometer_only_LHS',
    #     r'SWIR3 ($N_{\lambda}=801)$': 'synthetic_remotap_swir3_ratio_spectrometer_only_LHS',
    # }
    # data_dict = {
    #     r'NIR ($N_{\lambda}=211$)': 'synthetic_remotap_nir_spectrometer_only_LHS_matched_retrieval_era5',
    #     r'SWIR1 ($N_{\lambda}=851$)': 'synthetic_remotap_swir1_spectrometer_only_LHS_matched_retrieval_era5',
    #     r'SWIR3 ($N_{\lambda}=801)$': 'synthetic_remotap_swir3_spectrometer_only_LHS_matched_retrieval_era5',
    # }
    data_dict = {
        r'NIR ($N_{\lambda}=211$)': 'synthetic_remotap_nir_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper',
        r'SWIR1 ($N_{\lambda}=851$)': 'synthetic_remotap_swir1_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper',
        r'SWIR3 ($N_{\lambda}=801)$': 'synthetic_remotap_swir3_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper',
    }
    
    data_path = '/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/output/datasets'
    figure_path = '/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/output/figures'
    os.makedirs(figure_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,3))

    df = {'band': [], 'n_components': [], 'rel_rmse': [], 'rel_bias': [], 'explained_var': []}

    for band, dataset_name in data_dict.items():
        print(f'Processing {band} band...')

        train_path = os.path.join(data_path, f'{dataset_name}_train_val')
        train_dataset = torch.load(os.path.join(train_path, 'train_dataset.pt'), weights_only=False)
        train_indices = torch.load(os.path.join(train_path, 'train_indices.pt'), weights_only=False)
        val_indices = torch.load(os.path.join(train_path, 'val_indices.pt'), weights_only=False)

        n_subset = int(train_frac * len(train_indices))
        train_indices = np.random.choice(train_indices, n_subset, replace=False)

        # targets = test_dataset.targets.numpy()
        train_targets = train_dataset.targets.numpy()[train_indices]
        val_targets = train_dataset.targets.numpy()[val_indices]

        errors = []
        # n_max = 20 # maximum number of PCA components to test
        n_max = train_targets.shape[1]
        n_range = np.arange(0, n_max, 10)
        n_range[0] = 1 # we need at least one component
        for n in n_range:
            pca = PCA(n_components=n)

            # transform training data and fit PCA
            norm_data, log_mean, log_std = transform_targets(train_targets)
            pca.fit(norm_data)

            # get explained variance
            explained_var = np.sum(pca.explained_variance_ratio_) * 100

            # evaluate reconstruction error on validation set
            err = reconstruction_error(pca, val_targets, log_mean, log_std, z=normalization)
            bias = reconstruction_bias(pca, val_targets, log_mean, log_std, z=normalization)
            print(f'  n_components={n}, rel_rmse={err:.4f}%, explained_var={explained_var:.4f}%')
            df['band'].append(band)
            df['n_components'].append(n)
            df['rel_rmse'].append(err)
            df['rel_bias'].append(bias)
            df['explained_var'].append(explained_var)

            errors.append(reconstruction_error(pca, val_targets, log_mean, log_std))

        ax.plot(n_range, errors, label=band)

    ax.set_yscale('log')
    ax.set_xlabel('number of PCA components')
    ax.set_ylabel('relative RMSE [%]')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    fig.savefig(os.path.join(figure_path, f'{dataset_type}_pca_reconstruction_error.png'), dpi=300, bbox_inches='tight')

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(figure_path, f'{dataset_type}_pca_reconstruction_error.csv'), index=False)