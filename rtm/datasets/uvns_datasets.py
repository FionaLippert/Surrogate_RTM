import os
import os.path as osp
from datetime import datetime
from collections import OrderedDict
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from typing import Union, List, Dict, Tuple, Callable
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import Compose
from scipy.interpolate import interp1d

from rtm import rtm_transforms
from rtm.rtm_transforms import RTMTransformCollection, RTMTransform


class RTDataset(Dataset):
    """Synthetic measurements generated with RemoTAP forward model"""
    def __init__(self, 
                 data_dir: str,
                 prefix: str,
                 pixel_features: Union[str, List[str]] = 'all',
                 profile_features: Union[str, List[str]] = 'all',
                 target_quantity: str = 'radiance',
                 predict_difference: bool = False,
                 predict_ratio: bool = False,
                #  pixel_transform: transforms.RTMTransform = None,
                #  profile_transform: transforms.RTMTransform = None,
                #  target_transform: transforms.RTMTransform = None,
                #  proxy_transform: transforms.RTMTransform = None,
                 seed: int = 0,
                 train: bool = True,
                 test_size: float = 0.2, # TODO: explicitly define train/val/test split
                #  split_sizes: Dict[str, float] = {'train': 0.7, 'val': 0.1, 'test': 0.2},
                 radiance_thr: float = None,
                 include_jacobian: bool = False,
                 **kwargs
                 ):
        
        super().__init__()
        
        # define inputs and outputs
        self.pixel_features = pixel_features
        self.profile_features = profile_features 
        self.target_quantity = target_quantity
        self.predict_difference = predict_difference
        self.predict_ratio = predict_ratio
        self.include_jacobian = include_jacobian

        # transformations
        # self.pixel_transform = pixel_transform
        # self.profile_transform = profile_transform
        # self.target_transform = target_transform
        # self.proxy_transform = proxy_transform

        # collect all relevant data and identify dimensions
        self.pixel_data = self._collect_pixel_data(data_dir, prefix)
        self.n_pixel_features = self.pixel_data.shape[-1] if self.pixel_data is not None else 0
        self.profile_data = self._collect_profile_data(data_dir, prefix)
        self.n_profile_features = self.profile_data.shape[-1] if self.profile_data is not None else 0
        self.n_levels = self.profile_data.shape[1] if self.profile_data is not None else 0
        self.targets, self.proxy = self._collect_spectra(data_dir, prefix)
        self.wavelengths = self._load_wavelengths(data_dir, prefix)
        self.n_wavelengths = len(self.wavelengths)
        self.radiance_thr = radiance_thr

        if self.include_jacobian:
            # Collect jacobians for target and proxy spectra.
            # Here we assume that the jacobians have been precomputed for exactly the 
            # input features defined here and used by the NN.
            self.target_jacobians, self.proxy_jacobians = self._collect_jacobians(data_dir, prefix)
            self.gas_priors, self.gas_indices = self._collect_gas_priors(data_dir, prefix)
        else:
            self.target_jacobians, self.proxy_jacobians = None, None
            self.gas_priors, self.gas_indices = None, None
        # if target jacobians are available, compute SVD (only for selected input features)
        self.target_U, self.target_S, self.target_Vt = self._compute_jacobian_svd(self.target_jacobians) 

        # fix random seed for reproducibility
        self.seed = seed

        # optional: filter out large radiances
        if radiance_thr is not None:
            continuum = self.targets.max(1)
            self.indices = np.where(continuum < radiance_thr)[0]
            p_filtered = self.targets.shape[0] - len(self.indices)
            print(f'Filtered out {p_filtered} pixels with continuum above {self.radiance_thr}')
        else:
            self.indices = np.arange(self.targets.shape[0])
        

        # define train/test split
        self.train = train
        train_indices, test_indices = self.get_data_split(test_size)
        self.train_indices = self.indices[train_indices]
        self.test_indices = self.indices[test_indices]
        self.indices = self.train_indices if self.train else self.test_indices


        # prepare final data
        if self.pixel_data is not None: 
            self.pixel_data = self.pixel_data[self.indices]
            self.pixel_data = rtm_transforms.numpy_to_tensor(self.pixel_data)

        if self.profile_data is not None: 
            self.profile_data = self.profile_data[self.indices]
            self.profile_data = rtm_transforms.numpy_to_tensor(self.profile_data)

        self.targets = self.targets[self.indices]
        self.targets = rtm_transforms.numpy_to_tensor(self.targets)

        if self.proxy is not None:
            self.proxy = self.proxy[self.indices]
            self.proxy = rtm_transforms.numpy_to_tensor(self.proxy)

        if self.target_jacobians is not None:
            self.target_jacobians = self.target_jacobians[self.indices]
            self.target_jacobians = rtm_transforms.numpy_to_tensor(self.target_jacobians)

            # also prepare SVD components
            self.target_U = self.target_U[self.indices]
            self.target_S = self.target_S[self.indices]
            self.target_Vt = self.target_Vt[self.indices]
            self.target_U = rtm_transforms.numpy_to_tensor(self.target_U)
            self.target_S = rtm_transforms.numpy_to_tensor(self.target_S)
            self.target_Vt = rtm_transforms.numpy_to_tensor(self.target_Vt)

            # also prepare gas indices
            if len(self.gas_indices) > 0:
                self.gas_priors = self.gas_priors[self.indices]
                self.gas_priors = rtm_transforms.numpy_to_tensor(self.gas_priors)
                self.gas_indices = rtm_transforms.numpy_to_tensor(self.gas_indices, dtype=torch.int)

        if self.proxy_jacobians is not None:
            self.proxy_jacobians = self.proxy_jacobians[self.indices]
            self.proxy_jacobians = rtm_transforms.numpy_to_tensor(self.proxy_jacobians)

        if self.predict_difference:
            if self.proxy is None:
                raise ValueError('Proxy spectra are required to predict differences')
            else:
                self.targets = self.targets - self.proxy
        elif self.predict_ratio:
            if self.proxy is None:
                raise ValueError('Proxy spectra are required to predict ratios')
            else:
                self.targets = self.targets / self.proxy


    # def add_transforms(self, 
    #                    pixel_transform=None, 
    #                    profile_transform=None, 
    #                    target_transform=None,
    #                    proxy_transform=None):
    #     # add new transforms to existing transforms
    #     if pixel_transform:
    #         self.pixel_transform = transforms.RTMCompose([self.pixel_transform, pixel_transform]) \
    #             if self.pixel_transform else pixel_transform
    #     if profile_transform:
    #         self.profile_transform = transforms.RTMCompose([self.profile_transform, profile_transform]) \
    #             if self.profile_transform else profile_transform
    #     if target_transform:
    #         self.target_transform = transforms.RTMCompose([self.target_transform, target_transform]) \
    #             if self.target_transform else target_transform
    #     if proxy_transform:
    #         self.proxy_transform = transforms.RTMCompose([self.proxy_transform, proxy_transform]) \
    #             if self.proxy_transform else proxy_transform
            
    # def get_transforms(self):
    #     return self.pixel_transform, self.profile_transform, self.target_transform, self.proxy_transform
    
    # def set_transforms(self, 
    #                    pixel_transform=None, 
    #                    profile_transform=None, 
    #                    target_transform=None,
    #                    proxy_transform=None):
    #     # overwrite existing transforms
    #     if pixel_transform is not None:
    #         self.pixel_transform = pixel_transform
    #     if profile_transform is not None:
    #         self.profile_transform = profile_transform
    #     if target_transform is not None:
    #         self.target_transform = target_transform
    #     if proxy_transform is not None:
    #         self.proxy_transform = proxy_transform
            
    # def add_normalization(self, 
    #                       indices: List[int] = None, 
    #                       **kwargs):
    #     # add normalization to existing transforms
    #     self.add_transforms(*self.normalization(indices, **kwargs))
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):

        # get pixel data and apply transform if available
        if self.pixel_data is not None:
            x_pixels = self.pixel_data[idx]
            # if self.pixel_transform:
            #     x_pixels = self.pixel_transform(x_pixels)
        else:
            x_pixels = None

        # get profile data and apply transform if available
        if self.profile_data is not None:
            x_profiles = self.profile_data[idx]
            # if self.profile_transform:
            #     x_profiles = self.profile_transform(x_profiles)
        else:
            x_profiles = None

        # get target data and apply transform if available
        y = self.targets[idx]
        # if self.target_transform:
        #     y = self.target_transform(y)

        data_dict = {
            'inputs': x_pixels,
            'profile_inputs': x_profiles,
            'targets': y,
        }

        # get proxy data and apply transform if available
        if self.proxy is not None:
            proxy = self.proxy[idx]
            # if self.proxy_transform:
            #     proxy = self.proxy_transform(proxy)
            data_dict['proxy_targets'] = proxy

        if self.target_jacobians is not None:
            jac = self.target_jacobians[idx]
            data_dict['target_jacobians'] = jac

            # also include SVD components
            data_dict['target_U'] = self.target_U[idx] # shape (nwave, min(nwave, nfeatures))
            data_dict['target_S'] = self.target_S[idx] # shape (min(nwave, nfeatures),)
            data_dict['target_Vt'] = self.target_Vt[idx] # shape (min(nwave, nfeatures), nfeatures)

            if self.gas_priors is not None:
                data_dict['gas_priors'] = self.gas_priors[idx]
                data_dict['gas_indices'] = self.gas_indices

        if self.proxy_jacobians is not None:
            proxy_jacobian = self.proxy_jacobians[idx]
            data_dict['proxy_jacobians'] = proxy_jacobian


        return data_dict
    
    def _collect_pixel_data(self, data_dir: str, prefix: str):

        if self.pixel_features is None:
            # Nothing to be done here
            return None
        
        # load pixel netcdf file
        pixel_file = f'{prefix}_pixel_data.nc'
        print(f'Loading pixel data from {osp.join(data_dir, pixel_file)}')
        ds = xr.load_dataset(osp.join(data_dir, pixel_file))

        # extract relevant features
        if self.pixel_features == 'all':
            self.pixel_features = list(ds.data_vars.keys())    

        # get number of pixels
        n_pixels = ds.dims['pixelID']
        
        self.pixel_feature_names = []
        pixel_data = []
        for feature in self.pixel_features:
            # get data from netcdf dataset
            data = ds[feature].data.reshape(n_pixels, -1)

            # make sure data is float64 to avoid precision issues when normalizing
            data = data.astype(np.float64)
            pixel_data.append(data)

            f_dim = data.shape[-1]
            if f_dim == 1:
                self.pixel_feature_names.append(feature)
            else:
                self.pixel_feature_names.extend([f'{feature}_{i}' for i in range(f_dim)])

        if len(pixel_data) > 0:
            pixel_data = np.concatenate(pixel_data, axis=-1)
        else:
            pixel_data = None
          
        return pixel_data # shape: (nobs, nvars)
    
    def _collect_profile_data(self, data_dir: str, prefix: str):
        
        if self.profile_features is None:
            # Nothing to be done here
            return None

        # load profile netcdf file
        profile_file = f'{prefix}_profile_data.nc'
        print(f'Loading profile data from {osp.join(data_dir, profile_file)}')
        ds = xr.load_dataset(osp.join(data_dir, profile_file))

        # extract relevant features
        if self.profile_features == 'all':
            self.profile_features = list(ds.data_vars.keys())

        self.profile_feature_names = []
        profile_data = []
        for feature in self.profile_features:
            data = ds[feature].data
            # make sure data is float64 to avoid precision issues when normalizing
            data = data.astype(np.float64)
            profile_data.append(data)

            f_dim = data.shape[-1]
            self.profile_feature_names.extend([f'{feature}_{i}' for i in range(f_dim)])
        
        if len(profile_data) > 0:
            profile_data = np.stack(profile_data, axis=-1)
        else:
            profile_data = None
        
        return profile_data # shape: (nobs, nlevels, nvars)
    
    def _collect_spectra(self, data_dir: str, prefix: str):
        # load spectra netcdf file
        spectra_file = f'{prefix}_spectra.nc'
        print(f'Loading spectra from {osp.join(data_dir, spectra_file)}')
        ds = xr.load_dataset(osp.join(data_dir, spectra_file))

        # extract full physics spectra
        full_physics = ds[self.target_quantity].data # shape (nobs, nwave)

        # make sure data is float64 to avoid precision issues when normalizing
        full_physics = full_physics.astype(np.float64)

        # extract proxy spectra, if available
        if 'proxy' in ds.data_vars:
            proxy = ds['proxy'].data # shape (nobs, nwave)
            proxy = proxy.astype(np.float64)
        else:
            proxy = None
        
        return full_physics, proxy
    
    def _collect_jacobians(self, data_dir: str, prefix: str):

        # For now, we assume only pixel features are used as inputs to the NN.
        if self.pixel_features is None:
            return None, None

        # load jacobian netcdf file
        jacobian_file = f'{prefix}_jacobians.nc'
        print(f'Loading jacobians from {osp.join(data_dir, jacobian_file)}')
        ds = xr.load_dataset(osp.join(data_dir, jacobian_file))

        # extract jacobians only for selected pixel features
        # (assuming that _collect_pixel_data has been called before)
        jacobian_data = []
        proxy_jacobian_data = []
        for feature in self.pixel_feature_names:
            data = ds[f'{feature}_jac'].data
            # make sure data is float64 to avoid precision issues when normalizing
            data = data.astype(np.float64)
            jacobian_data.append(data)

            if f'{feature}_proxy_jac' in ds.data_vars:
                # also load the proxy jacobian for this feature
                proxy_data = ds[f'{feature}_proxy_jac'].data
                proxy_data = proxy_data.astype(np.float64)
                proxy_jacobian_data.append(proxy_data)

        if len(jacobian_data) > 0:
            # stack along the last dimension to get shape (nobs, nwave, nvars)
            jacobian_data = np.stack(jacobian_data, axis=-1)
        else:
            jacobian_data = None

        if len(proxy_jacobian_data) > 0:
            # stack along the last dimension to get shape (nobs, nwave, nvars)
            proxy_jacobian_data = np.stack(proxy_jacobian_data, axis=-1)
        else:
            proxy_jacobian_data = None

        return jacobian_data, proxy_jacobian_data

    def _collect_gas_priors(self, data_dir: str, prefix: str):

        # For now, we assume only pixel features are used as inputs to the NN.
        if self.pixel_features is None:
            return None, None

        # load jacobian netcdf file
        jacobian_file = f'{prefix}_jacobians.nc'
        print(f'Loading gas priors from {osp.join(data_dir, jacobian_file)}')
        ds = xr.load_dataset(osp.join(data_dir, jacobian_file))

        # extract jacobians only for selected pixel features
        # (assuming that _collect_pixel_data has been called before)
        gas_priors = []
        gas_indices = []
        for fidx, feature in enumerate(self.pixel_feature_names):
            if feature.startswith('vmol_'):
                data = ds[f'{feature}_prior'].data
                # make sure data is float64 to avoid precision issues when normalizing
                data = data.astype(np.float64)
                gas_priors.append(data)
                gas_indices.append(fidx)

        if len(gas_priors) > 0:
            # stack along the last dimension to get shape (nobs, nwave, nvars)
            gas_priors = np.stack(gas_priors, axis=-1)
        else:
            gas_priors = None

        gas_indices = np.array(gas_indices, dtype=int)

        return gas_priors, gas_indices
    
    def _compute_jacobian_svd(self, jacobians: np.ndarray):

        if jacobians is None:
            return None, None, None
        
        # compute SVD of target jacobians
        U, S, Vt = np.linalg.svd(jacobians, full_matrices=False, compute_uv=True)

        return U, S, Vt

    
    def _load_wavelengths(self, data_dir: str, prefix: str):
        # load wavelengths from spectra netcdf file
        print(f'Loading wavelengths from {osp.join(data_dir, prefix+"_spectra.nc")}')
        spectra_file = f'{prefix}_spectra.nc'
        ds = xr.load_dataset(osp.join(data_dir, spectra_file))
        wavelengths = ds['wavelength'].data.astype(np.float64)
        return wavelengths
        
    
    def normalization(self, 
                      transforms: Dict[str, RTMTransformCollection],
                      indices: List[int] = None, 
                      per_wavelength: bool = True,
                      per_layer: bool = False,
                      per_state_variable: bool = True,
                      coupled_proxy: bool = False, 
                      max_spectral_scaling: bool = False,
                      **kwargs):
        # define indices to use to compute normalization 
        if indices is None: indices = range(len(self))

        print(f'transforms: {transforms}')

        pixel_stats = self.get_pixel_statistics(
            indices=indices, 
            transform=transforms.get('pixel_transform', None)
        )
        profile_stats = self.get_profile_statistics(
            indices=indices, 
            transform=transforms.get('profile_transform', None),
            per_layer=per_layer
        )
        target_stats = self.get_target_statistics(
            indices=indices, 
            transform=transforms.get('target_transform', None),
            per_wavelength=per_wavelength
        )
        proxy_stats = self.get_proxy_statistics(
            indices=indices, 
            transform=transforms.get('proxy_transform', None),
            per_wavelength=per_wavelength
        )
        jacobian_stats = self.get_jacobian_statistics(
            indices=indices, 
            transform=transforms.get('jacobian_transform', None),
            per_state_variable=per_state_variable
        )

        if max_spectral_scaling:
            # scale spectral data by overall maximum value
            target_stats['mean'] = 0.0
            target_stats['std'] = target_stats['max']
            proxy_stats['mean'] = 0.0
            proxy_stats['std'] = proxy_stats['max']

        pixel_norm = rtm_transforms.normalization(pixel_stats, ) if pixel_stats else None
        profile_norm = rtm_transforms.normalization(profile_stats) if profile_stats else None
        target_norm = rtm_transforms.normalization(target_stats)
        jacobian_norm = rtm_transforms.normalization(jacobian_stats) if jacobian_stats else None

        if coupled_proxy:
            proxy_norm = target_norm
        elif proxy_stats:
            proxy_norm = rtm_transforms.normalization(proxy_stats)
        else:
            proxy_norm = None

        return {
            'pixel_transform': pixel_norm, 
            'profile_transform': profile_norm, 
            'target_transform': target_norm,
            'proxy_transform': proxy_norm,
            'jacobian_transform': jacobian_norm
        }
    
    def get_pixel_statistics(self, indices: List[int], transform: RTMTransform = None):
        if self.pixel_data is not None:
            data = self.pixel_data[indices]
            if transform is not None:
                data = transform(data)
        else:
            return None

        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # compute stats
        stats = {
            'mean': data.mean(axis=0),
            'std': data.std(axis=0),
            'min': data.min(axis=0),
            'max': data.max(axis=0),
        }

        return stats
    
    def get_profile_statistics(self, 
                               indices: List[int], 
                               transform: RTMTransform = None,
                               per_layer: bool = False
                               ):
        if self.profile_data is not None:
            data = self.profile_data[indices]
            if transform is not None:
                data = transform(data)
        else:
            return None

        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # compute stats
        kwargs = dict(axis=0) if per_layer else dict(axis=(0, 1))
        stats = {
            'mean': data.mean(**kwargs),
            'std': data.std(**kwargs),
            'min': data.min(**kwargs),
            'max': data.max(**kwargs),
        }

        return stats
    
    def get_target_statistics(self, 
                              indices: List[int], 
                              transform: RTMTransform = None,
                              per_wavelength: bool = True
                              ):
        targets = self.targets[indices]
        print(f'target shape: {targets.shape}')
        if transform is not None:
            targets = transform(targets)
            print(f'target shape after transform: {targets.shape}')

        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        # compute stats
        kwargs = dict(axis=0) if per_wavelength else dict()
        stats = {
            'mean': targets.mean(**kwargs),
            'std': targets.std(**kwargs),
            'min': targets.min(**kwargs),
            'max': targets.max(**kwargs),
        }

        print('mean shape:', stats['mean'].shape)

        return stats
    
    def get_proxy_statistics(self, 
                             indices: List[int], 
                             transform: RTMTransform = None,
                             per_wavelength: bool = True
                             ):
        if self.proxy is not None:
            proxy = self.proxy[indices]
            if transform is not None:
                proxy = transform(proxy)
        else:
            return None

        if isinstance(proxy, torch.Tensor):
            proxy = proxy.numpy()

        # compute stats
        kwargs = dict(axis=0) if per_wavelength else dict()
        stats = {
            'mean': proxy.mean(**kwargs),
            'std': proxy.std(**kwargs),
            'min': proxy.min(**kwargs),
            'max': proxy.max(**kwargs),
        }

        return stats
    
    def get_jacobian_statistics(self, 
                             indices: List[int], 
                             transform: RTMTransform = None,
                             per_state_variable: bool = True
                             ):
        if hasattr(self, 'target_jacobians') and self.target_jacobians is not None:
            jacobian = self.target_jacobians[indices]
            if transform is not None:
                jacobian = transform(jacobian)
        else:
            return None

        if isinstance(jacobian, torch.Tensor):
            jacobian = jacobian.numpy()

        # compute stats
        kwargs = dict(axis=(0, 1)) if per_state_variable else dict()
        stats = {
            'mean': jacobian.mean(**kwargs),
            'std': jacobian.std(**kwargs),
            'min': jacobian.min(**kwargs),
            'max': jacobian.max(**kwargs),
        }

        return stats

    def get_data_split(self, test_size: float = 0.2):

        n_total = len(self)
        print(f'Total number of data points: {n_total}')
        n_test = int(test_size * n_total)
        n_train = n_total - n_test

        print(f'Train size: {n_train}, Test size: {n_test}')

        if n_train == 0:
            print('Warning: No training data available. Using unshuffled data for testing.')
            idx = np.arange(n_total)
        else:
            idx = np.random.RandomState(seed=self.seed).permutation(n_total)

        # split indices into train and test sets
        train_indices = idx[:n_train]
        test_indices = idx[n_train:]

        # train_set, test_set = random_split(self, [n_train, n_test], 
                            #   generator=torch.Generator().manual_seed(self.seed))
        
        return train_indices, test_indices
    

class FlatRTDataset(RTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # flatten profile data and concatenate with pixel data
        if self.n_profile_features > 0:
            n_flat = self.n_levels * self.n_profile_features
            profile_data = torch.permute(self.profile_data, (0, 2, 1))
            profile_data = profile_data.reshape(-1, n_flat)
            if self.pixel_data is not None:
                self.pixel_data = torch.cat([self.pixel_data, profile_data], dim=-1)
            else:
                self.pixel_data = profile_data

            self.n_pixel_features = self.n_pixel_features + n_flat
            self.pixel_feature_names = self.pixel_feature_names + self.profile_feature_names
    
    def __getitem__(self, idx: int):

        if self.pixel_data is not None:
            x = self.pixel_data[idx]
            # if self.pixel_transform:
                # x = self.pixel_transform(x)
        else:
            x = None

        y = self.targets[idx]
        # if self.target_transform:
            # y = self.target_transform(y)


        data_dict = {
            'inputs': x,
            'targets': y,
        }

        if self.proxy is not None:
            proxy = self.proxy[idx]
            # if self.proxy_transform:
                # proxy = self.proxy_transform(proxy)
            data_dict['proxy_targets'] = proxy

        if hasattr(self, 'target_jacobians') and self.target_jacobians is not None:
            jac = self.target_jacobians[idx]
            data_dict['target_jacobians'] = jac

            # also include SVD components
            data_dict['target_U'] = self.target_U[idx] # shape (nwave, min(nwave, nfeatures))
            data_dict['target_S'] = self.target_S[idx] # shape (min(nwave, nfeatures),)
            data_dict['target_Vt'] = self.target_Vt[idx] # shape (min(nwave, nfeatures), nfeatures)

            if self.gas_priors is not None:
                data_dict['gas_priors'] = self.gas_priors[idx]
                data_dict['gas_indices'] = self.gas_indices

        if hasattr(self, 'proxy_jacobians') and self.proxy_jacobians is not None:
            proxy_jacobian = self.proxy_jacobians[idx]
            data_dict['proxy_jacobians'] = proxy_jacobian

        return data_dict


    
