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

from rtm import utils

class RTDataset(Dataset):
    """Synthetic measurements generated with RemoTeC forward model"""
    def __init__(self, 
                 data_dir: str,
                 prefix: str = 'syn_meas_s5_random_ensemble_1e5_combined',
                 pixel_features: Union[str, List[str]] = 'all',
                 profile_features: Union[str, List[str]] = 'all',
                 target_quantity: str = 'radiance',
                 predict_difference: bool = False,
                 pixel_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 profile_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 target_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 proxy_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 seed: int = 0,
                 train: bool = True,
                 test_size: float = 0.2, # TODO: explicitly define train/val/test split
                #  split_sizes: Dict[str, float] = {'train': 0.7, 'val': 0.1, 'test': 0.2},
                 **kwargs
                 ):
        
        # define inputs and outputs
        self.pixel_features = pixel_features
        self.profile_features = profile_features 
        self.target_quantity = target_quantity
        self.predict_difference = predict_difference

        # transformations
        self.pixel_transform = pixel_transform
        self.profile_transform = profile_transform
        self.target_transform = target_transform
        self.proxy_transform = proxy_transform

        # collect all relevant data and identify dimensions
        self.pixel_data = self._collect_pixel_data(data_dir, prefix)
        self.n_pixel_features = self.pixel_data.shape[-1] if self.pixel_data is not None else 0
        self.profile_data = self._collect_profile_data(data_dir, prefix)
        self.n_profile_features = self.profile_data.shape[-1] if self.profile_data is not None else 0
        self.n_levels = self.profile_data.shape[1] if self.profile_data is not None else 0
        self.targets, self.proxy = self._collect_spectra(data_dir, prefix)
        self.wavelengths = self._load_wavelengths(data_dir, prefix)
        self.n_wavelengths = len(self.wavelengths)
        self.indices = np.arange(self.targets.shape[0])

        # fix random seed for reproducibility
        self.seed = seed

        # define train/test split
        self.train = train
        self.train_indices, self.test_indices = self.get_data_split(test_size)
        self.indices = self.train_indices if self.train else self.test_indices

        if self.pixel_data is not None: self.pixel_data = self.pixel_data[self.indices]
        if self.profile_data is not None: self.profile_data = self.profile_data[self.indices]
        self.targets = self.targets[self.indices]
        self.proxy = self.proxy[self.indices] if self.proxy is not None else None

        if self.predict_difference:
            self.targets = self.targets - self.proxy


    def add_transforms(self, 
                       pixel_transform=None, 
                       profile_transform=None, 
                       target_transform=None,
                       proxy_transform=None):
        # add new transforms to existing transforms
        if pixel_transform:
            self.pixel_transform = utils.RTMCompose([self.pixel_transform, pixel_transform]) \
                if self.pixel_transform else pixel_transform
        if profile_transform:
            self.profile_transform = utils.RTMCompose([self.profile_transform, profile_transform]) \
                if self.profile_transform else profile_transform
        if target_transform:
            self.target_transform = utils.RTMCompose([self.target_transform, target_transform]) \
                if self.target_transform else target_transform
        if proxy_transform:
            self.proxy_transform = utils.RTMCompose([self.proxy_transform, proxy_transform]) \
                if self.proxy_transform else proxy_transform
            
    def get_transforms(self):
        return self.pixel_transform, self.profile_transform, self.target_transform, self.proxy_transform
    
    def set_transforms(self, 
                       pixel_transform=None, 
                       profile_transform=None, 
                       target_transform=None,
                       proxy_transform=None):
        # overwrite existing transforms
        if pixel_transform is not None:
            self.pixel_transform = pixel_transform
        if profile_transform is not None:
            self.profile_transform = profile_transform
        if target_transform is not None:
            self.target_transform = target_transform
        if proxy_transform is not None:
            self.proxy_transform = proxy_transform
            
    def add_normalization(self, 
                          indices: List[int] = None, 
                          **kwargs):
        self.add_transforms(*self.normalization(indices, **kwargs))
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):

        # get pixel data and apply transform if available
        if self.pixel_data is not None:
            x_pixels = self.pixel_data[idx]
            if self.pixel_transform:
                x_pixels = self.pixel_transform(x_pixels)
        else:
            x_pixels = None

        # get profile data and apply transform if available
        if self.profile_data is not None:
            x_profiles = self.profile_data[idx]
            if self.profile_transform:
                x_profiles = self.profile_transform(x_profiles)
        else:
            x_profiles = None

        # get target data and apply transform if available
        y = self.targets[idx]
        if self.target_transform:
            y = self.target_transform(y)

        data_dict = {
            'inputs': x_pixels,
            'profile_inputs': x_profiles,
            'targets': y,
        }

        # get proxy data and apply transform if available
        if self.proxy is not None:
            proxy = self.proxy[idx]
            if self.proxy_transform:
                proxy = self.proxy_transform(proxy)
            data_dict['proxy_targets'] = proxy

        return data_dict
    
    def _collect_pixel_data(self, data_dir: str, prefix: str):
        # load pixel netcdf file
        pixel_file = f'{prefix}_pixel_data.nc'
        ds = xr.open_dataset(osp.join(data_dir, pixel_file))

        # get number of pixels
        n_pixels = ds.dims['pixelID']

        # extract relevant features
        if self.pixel_features == 'all':
            self.pixel_features = list(ds.data_vars.keys())
        elif self.pixel_features is None:
            return None
        
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
        # load profile netcdf file
        profile_file = f'{prefix}_profile_data.nc'
        ds = xr.open_dataset(osp.join(data_dir, profile_file))

        # extract relevant features
        if self.profile_features == 'all':
            self.profile_features = list(ds.data_vars.keys())
        elif self.profile_features is None:
            return None

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
        ds = xr.open_dataset(osp.join(data_dir, spectra_file))

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
    
    def _load_wavelengths(self, data_dir: str, prefix: str):
        # load wavelengths from spectra netcdf file
        spectra_file = f'{prefix}_spectra.nc'
        ds = xr.open_dataset(osp.join(data_dir, spectra_file))
        wavelengths = ds['wavelength'].data.astype(np.float64)
        return wavelengths
        
    
    def normalization(self, 
                      indices: List[int] = None, 
                      per_wavelength: bool = True,
                      per_layer: bool = False,
                      coupled_proxy: bool = False, 
                      **kwargs):
        if indices is None: indices = range(len(self))
        pixel_stats = self.get_pixel_statistics(indices)
        profile_stats = self.get_profile_statistics(indices, per_layer)
        target_stats = self.get_target_statistics(indices, per_wavelength)
        proxy_stats = self.get_proxy_statistics(indices, per_wavelength)
        pixel_norm = utils.normalization(pixel_stats) if pixel_stats else None
        profile_norm = utils.normalization(profile_stats) if profile_stats else None
        target_norm = utils.normalization(target_stats)
        proxy_norm = utils.normalization(proxy_stats)

        if coupled_proxy:
            proxy_norm = target_norm

        return pixel_norm, profile_norm, target_norm, proxy_norm
    
    def get_pixel_statistics(self, indices: List[int]):
        if self.pixel_transform and self.pixel_data is not None:
            data = self.pixel_transform(self.pixel_data[indices])
        elif self.pixel_data is not None:
            data = self.pixel_data[indices]
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
    
    def get_profile_statistics(self, indices: List[int], per_layer: bool = False):
        if self.profile_transform and self.profile_data is not None:
            data = self.profile_transform(self.profile_data[indices])
        elif self.profile_data is not None:
            data = self.profile_data[indices]
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
    
    def get_target_statistics(self, indices: List[int], per_wavelength: bool = True):
        if self.target_transform:
            targets = self.target_transform(self.targets[indices])
        else:
            targets = self.targets[indices]

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

        return stats
    
    def get_proxy_statistics(self, indices: List[int], per_wavelength: bool = True):
        if self.proxy_transform and self.proxy is not None:
            proxy = self.proxy_transform(self.proxy[indices])
        elif self.proxy is not None:
            proxy = self.proxy[indices]
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

    def get_data_split(self, test_size: float = 0.2):

        n_total = len(self)
        n_test = int(test_size * n_total)
        n_train = n_total - n_test

        perm = np.random.RandomState(seed=self.seed).permutation(n_total)

        train_indices = perm[:n_train]
        test_indices = perm[n_train:]

        # train_set, test_set = random_split(self, [n_train, n_test], 
                            #   generator=torch.Generator().manual_seed(self.seed))
        
        return train_indices, test_indices
    

class FlatRTDataset(RTDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # flatten profile data and concatenate with pixel data
        if self.n_profile_features > 0:
            n_flat = self.n_levels * self.n_profile_features
            profile_data = np.transpose(self.profile_data, (0, 2, 1))
            profile_data = profile_data.reshape(-1, n_flat)
            if self.pixel_data is not None:
                self.pixel_data = np.concatenate([self.pixel_data, profile_data], axis=-1)
            else:
                self.pixel_data = profile_data

            self.n_pixel_features = self.n_pixel_features + n_flat
            self.pixel_feature_names = self.pixel_feature_names + self.profile_feature_names
    
    def __getitem__(self, idx: int):

        if self.pixel_data is not None:
            x = self.pixel_data[idx]
            if self.pixel_transform:
                x = self.pixel_transform(x)
        else:
            x = None

        y = self.targets[idx]
        if self.target_transform:
            y = self.target_transform(y)


        data_dict = {
            'inputs': x,
            'targets': y,
        }

        if self.proxy is not None:
            proxy = self.proxy[idx]
            if self.proxy_transform:
                proxy = self.proxy_transform(proxy)
            data_dict['proxy_targets'] = proxy

        return data_dict


    
