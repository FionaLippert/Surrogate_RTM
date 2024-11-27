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

class RadiativeTransferDataset(Dataset):
    """Synthetic measurements generated with LINTRAN forward model"""
    def __init__(self, 
                 data_dir: str, 
                 prefix: str,
                 band: str = 'BAND4',
                 input_vars: Union[str, List[str]] = 'all',
                 wavelength_idx: Union[str, List[float]] = 'all',
                 target_var: str = 'radiance',
                 months: List[int] = range(1, 13),
                 transform: utils.RTMTransform = None,
                 target_transform: utils.RTMTransform = None,
                 seed: int = 0,
                 **kwargs
                 ):
        
        # TODO: construct dataset separately for train and test split?
        # for test split, automatically normalize based on training data
        
        self.data_dir = data_dir
        self.band = band
        self.input_vars = input_vars
        self.target_var = target_var
        self.wavelength_idx = wavelength_idx
        self.transform = transform
        self.target_transform = target_transform

        # time, longitude and latitude (used for selecting subsets of data)
        self.coords = load_coords(data_dir, prefix)
        self.wavelengths = load_wavelengths(data_dir, prefix, band)

        # collect all relevant data
        self.data = {}
        self.data.update(load_pixel_data(data_dir, prefix, band))
        self.data.update(load_profile_data(data_dir, prefix))
        self.data.update(load_spectra(data_dir, prefix, band, wavelength=False))

        self.data['solar_radiance'] = self.data['irradiance'] / np.pi * \
                                      self.data['cos_solar_zenith_angle'].reshape(-1, 1)

        # filter data by month
        self.mask = self.coords['time'].month.isin(months)
        self.coords = {k: v[self.mask] for k, v in self.coords.items()}
        
        self.data = {k: v[self.mask] for k, v in self.data.items()}

        # collect inputs
        self.inputs = self._dict_to_array(self._collect_inputs())

        # collect targets
        self.targets = self.data[self.target_var]
        if self.wavelength_idx != 'all':
            self.targets = self.targets[:, self.wavelength_idx]
            self.wavelengths = self.wavelengths[self.wavelength_idx]

        self.seed = seed

    def set_transforms(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def add_transforms(self, transform=None, target_transform=None):
        if transform:
            self.transform = utils.RTMCompose([self.transform, transform]) if self.transform else transform
        if target_transform:
            self.target_transform = utils.RTMCompose([self.target_transform, target_transform]) \
                if self.target_transform else target_transform
            
    def get_var(self, var: str) -> np.ndarray:
        return self.data[var]
    
    def __len__(self):
        return self.mask.sum()
    
    @property
    def indices(self):
        return [i for i in range(len(self))]

    def __getitem__(self, idx: int):

        x = self.inputs[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        data_dict = {
            'inputs': x,
            'targets': y,
            'wavelengths': self.wavelengths,
            'solar_radiance': self.data['solar_radiance'][idx],
            'surface_albedo': self.data['surface_albedo'][idx],
            'air_mass_factor': self.data['air_mass_factor'][idx],
        }

        return data_dict
        
    def _collect_inputs(self):
        if self.input_vars == 'all':
            # use all available input variables
            return self.data
        else:
            return {k: v for k, v in self.data.items() if k in self.input_vars}
    
    def _dict_to_array(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        # convert dict of arrays to stacked numpy array

        # some quantities are one-dimensional (nobs), others are two-dimensional (nobs, nlevels)
        # so we need to expand dimensions of one-dimensional quantities
        d = {k: np.expand_dims(v, axis=-1) if v.ndim == 1 else v for k, v in d.items()}

        # store variable names and shapes
        self.input_shapes = OrderedDict({k: v.shape[-1] for k, v in d.items()})

        # now we can concatenate input variables along last axis
        a = np.concatenate(list(d.values()), axis=-1)
        return a
    
    def normalization(self, indices: List[int], per_wavelength: bool = True):
        input_stats = self.get_input_statistics(indices)
        target_stats = self.get_target_statistics(indices, per_wavelength)
        input_norm = utils.normalization(input_stats)
        target_norm = utils.normalization(target_stats)

        return input_norm, target_norm
    
    def get_input_statistics(self, indices: List[int]):
        if self.transform:
            inputs = self.transform(self.inputs[indices])
        else:
            inputs = self.inputs[indices]

        stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
        }

        idx = 0
        for name, dim in self.input_shapes.items():
            # print(name, dim)
            # get the subset of data for the current variable
            var_i = inputs[:, idx:idx+dim]
            if isinstance(var_i, torch.Tensor):
                var_i = var_i.numpy()
            idx += dim

            # compute stats and expand to the shape of the variable
            stats['mean'].append(np.broadcast_to(var_i.mean(), dim))
            stats['std'].append(np.broadcast_to(var_i.std(), dim))
            stats['min'].append(np.broadcast_to(var_i.min(), dim))
            stats['max'].append(np.broadcast_to(var_i.max(), dim))

        # concatenate mean and std along last axis
        for k, v in stats.items():
            stats[k] = np.concatenate(v, axis=-1)

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

    def get_data_split(self, test_size: float = 0.2):

        n_total = len(self)
        n_test = int(test_size * n_total)
        n_train = n_total - n_test

        train_set, test_set = random_split(self, [n_train, n_test], 
                              generator=torch.Generator().manual_seed(self.seed))
        
        return train_set, test_set
    


class RadiativeTransferVPDataset(RadiativeTransferDataset):
    """
    Synthetic measurements generated with LINTRAN forward model
    
    Atmospheric state is represented by vertical profiles of 
    trace gases and meteo variables.
    """
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        # number of vertical levels
        self.nlevels = self.inputs.shape[1]

    # overwrite method to collect inputs
    def _dict_to_array(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        # convert dict of arrays to stacked numpy array

        # some quantities are one-dimensional (nobs), others are two-dimensional (nobs, nlevels)
        # so we repeat one-dimensional quantities along vertical levels
        d = {k: utils.global_to_profile(v) if v.ndim == 1 else v for k, v in d.items()}

        # now we can stack input variables along new axis
        a = np.stack(list(d.values()), axis=-1) # shape (nobs, nlevels, nvars)
        return a

    # overwrite method to compute input statistics
    def get_input_statistics(self, indices: List[int]):
        if self.transform:
            inputs = self.transform(self.inputs[indices])
        else:
            inputs = self.inputs[indices]

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.numpy()

        # compute stats across samples and vertical levels
        stats = {
            'mean': inputs.mean(axis=(0, 1)),
            'std': inputs.std(axis=(0, 1)),
            'min': inputs.min(axis=(0, 1)),
            'max': inputs.max(axis=(0, 1)),
        }

        return stats
    
    
def prepare_data(target_dir: str,
                 data_dir: str,
                 prefix: str,
                 band: str,
                 gas_names: List[str] = ['co2', 'ch4', 'co'],
                ):
    
    os.makedirs(target_dir, exist_ok=True)

    # load pixel data and save to csv file
    pixel_data = load_pixel_data(data_dir, prefix, band, gas_names)
    pixel_data_to_netcdf(pixel_data, target_dir, prefix)

    # load profile data and save to csv file
    profile_data = load_profile_data(data_dir, prefix, gas_names)
    profile_data_to_netcdf(profile_data, target_dir, prefix)

    # load spectral data and save to csv files
    spectral_data = load_spectra(data_dir, prefix, band)
    spectra_to_netcdf(spectral_data, target_dir, prefix)

    # load coordinates and save to csv file
    coords = load_coords(data_dir, prefix)
    coords_to_netcdf(coords, target_dir, prefix)


def pixel_data_to_netcdf(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format and write to disk
    for k, v in data.items():
        data[k] = {'dims': ('pixelID'), 'data': v}
    data['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    data = xr.Dataset.from_dict(data)

    data.to_netcdf(osp.join(target_dir, f'{prefix}_pixel_data.nc'), format='NETCDF4')

def pixel_data_to_csv(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # save pixel data to csv file
    df_pixels = pd.DataFrame(data)
    df_pixels['pixelID'] = df_pixels.index
    df_pixels = df_pixels.to_csv(osp.join(target_dir, f'{prefix}_pixel_data.csv'), index=False)

def coords_to_netcdf(coords: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format and write to disk
    for k, v in coords.items():
        coords[k] = {'dims': ('pixelID'), 'data': v}
    coords['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    coords = xr.Dataset.from_dict(coords)

    coords.to_netcdf(osp.join(target_dir, f'{prefix}_coords.nc'), format='NETCDF4')

def coords_to_csv(coords: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # save coordinates to csv file
    df_coords = pd.DataFrame(coords)
    df_coords['pixelID'] = df_coords.index
    df_coords.to_csv(osp.join(target_dir, f'{prefix}_coords.csv'), index=False)


def profile_data_to_netcdf(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format
    for k, v in data.items():
        data[k] = {'dims': ('pixelID', 'layerID'), 'data': v}
    data['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    data['layerID'] = {'dims': ('layerID'), 'data': np.arange(v.shape[1])}
    data = xr.Dataset.from_dict(data)

    data.to_netcdf(osp.join(target_dir, f'{prefix}_profile_data.nc'), format='NETCDF4')

def profile_data_to_csv(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # save profile data to csv file
    # saved in long format, i.e. data is flattened and layer/pixel info is included in layerID and pixelID columns
    df_profiles = pd.DataFrame()

    shape = None
    for k, v in data.items():
        # check data shape: should be (nobs, nlayers)
        if shape is None :
            shape = v.shape
        else:
            assert shape == v.shape
        df_profiles[k] = v.flatten()
    nobs, nlayers = shape

    # save pixel and layer info
    df_profiles['pixelID'] = np.arange(nobs).repeat(nlayers)
    df_profiles['layerID'] = np.tile(np.arange(nlayers), nobs)

    # write to disk
    df_profiles.to_csv(osp.join(target_dir, f'{prefix}_profile_data.csv'), index=False)

def spectra_to_netcdf(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format
    for k, v in data.items():
        if k == 'wavelength':
            data[k] = {'dims': ('wavelength'), 'data': v}
        else:
            data[k] = {'dims': ('pixelID', 'wavelength'), 'data': v}
    data['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    data = xr.Dataset.from_dict(data)

    data.to_netcdf(osp.join(target_dir, f'{prefix}_spectra.nc'), format='NETCDF4')

def spectra_to_csv(data: Dict[str, np.ndarray], target_dir: str, prefix: str, quantity: str):
    # save spectra (only for the given quantity) to csv file
    df_spectra = pd.DataFrame(data[quantity], columns=data['wavelength'])
    df_spectra['pixelID'] = df_spectra.index
    df_spectra.to_csv(osp.join(target_dir, f'{prefix}_spectra.csv'), index=False)

def load_pixel_data(data_dir: str,
                      prefix: str,
                      band: str,
                      gas_names: List[str] = ['co2', 'ch4', 'co'],
                     ):
    # load pixel/surface properties that do not come as vertical profiles
    data = {}
    data.update(load_dem(data_dir, prefix))
    data.update(load_orbit(data_dir, prefix))
    data.update(load_sim_props(data_dir, prefix, band))
    data.update(load_surface_meteo(data_dir, prefix))
    data.update(load_aerosols(data_dir, prefix))
    data.update(load_cirrus(data_dir, prefix))

    # load trace gas concentrations
    data.update(load_trace_gas_total_columns(data_dir, prefix, gas_names))

    return data

def load_profile_data(data_dir: str,
                         prefix: str,
                         gas_names: List[str] = ['co2', 'ch4', 'co'],
                        ):
    # load vertical profiles of trace gases and meteo variables
    data = {}
    data.update(load_meteo_profiles(data_dir, prefix))
    data.update(load_trace_gas_profiles(data_dir, prefix, gas_names))
    
    # load aerosol profiles
    data_aer = load_aerosol_profiles(data_dir, prefix)

    # downsample all other profiles to match the aerosol pressure levels
    p_layers = data_aer['aer_p']
    p_high_res = data['p']
    for k, v in data.items():
        # unfortunately, neither numpy nor scipy seem to have a way to
        # interpolate along a single axis of a multi-dimensional array,
        # so we have to loop over all profiles
        data[k] = np.stack([np.interp(p_layers[i], p_high_res[i], v[i]) 
                            for i in range(v.shape[0])], axis=0)
    data.update(data_aer)

    return data


def load_dem(data_dir: str, prefix: str):
    # extract dem data from netCDF file
    dem_file = f'{prefix}_dem.nc'
    var_list = ['minh', 'maxh', 'meanh', 'stdh']
    ds_dem = xr.open_dataset(osp.join(data_dir, dem_file))
    dem = {var: ds_dem[var].data for var in var_list}
    return dem

def load_orbit(data_dir: str, prefix: str):
    # extract orbit data from netCDF file
    orbit_file = f'{prefix}_orbit.nc'
    var_list = [
        'solar_zenith_angle', 
        'viewing_zenith_angle', 
        'viewing_azimuth_angle', 
        'solar_azimuth_angle',
        ]
    ds_orbit = xr.open_dataset(osp.join(data_dir, orbit_file))
    orbit = {var: ds_orbit[var].data for var in var_list}

    orbit['relative_azimuth_angle'] = utils.azimuth_difference(
                                        orbit['viewing_azimuth_angle'],
                                        orbit['solar_azimuth_angle']
                                        )
    orbit_dict = {}
    for var, data in orbit.items():
        orbit_dict[var] = data
        if 'angle' in var:
            orbit_dict[f'cos_{var}'] = np.cos(np.deg2rad(data))
            orbit_dict[f'sin_{var}'] = np.sin(np.deg2rad(data))

    orbit_dict['air_mass_factor'] = (1 / orbit_dict['cos_solar_zenith_angle']) + \
                                    (1 / orbit_dict['cos_viewing_zenith_angle'])
    
    return orbit_dict


def load_coords(data_dir: str, prefix: str):
    orbit_file = f'{prefix}_orbit.nc'
    # extract datetime, longitude and latitude from netCDF file
    var_list = ['time', 'longitude', 'latitude']
    ds_coords = xr.open_dataset(osp.join(data_dir, orbit_file))
    coords = {var: ds_coords[var].data for var in var_list}

    # datetime is not read correctly by xarray, so load it with NetCDF4
    ds_coords = nc.Dataset(osp.join(data_dir, orbit_file))
    dt = pd.to_datetime([datetime(*ts) for ts in ds_coords['time'][:]])
    coords['time'] = dt

    return coords
        
def load_surface_meteo(data_dir: str, prefix: str):
    # extract surface meteo data from netCDF file
    meteo_file = f'{prefix}_meteo.nc'
    var_list = ['sp', 'z', 'v10', 'u10']
    ds_meteo = xr.open_dataset(osp.join(data_dir, meteo_file))
    meteo = {var: ds_meteo[var].data for var in var_list}

    # calculate total water vapor
    layers = {var: ds_meteo[var].data for var in ['hyai', 'hybi']}
    tpw = utils.total_column_amount(ds_meteo['q'].data, meteo['sp'], layers['hyai'], layers['hybi'])
    meteo['H2O_column_total'] = tpw
    meteo['log_H2O_column_total'] = np.log(tpw)

    return meteo

def load_meteo_profiles(data_dir: str, prefix: str):
    # extract verical meteo profiles from netCDF file
    meteo_file = f'{prefix}_meteo.nc'
    var_list = ['q', 't']
    ds_meteo = xr.open_dataset(osp.join(data_dir, meteo_file))
    meteo = {var: ds_meteo[var].data for var in var_list}
    layers = {var: ds_meteo[var].data for var in ['hyai', 'hybi']}

    # rename specific humidity q to H2O
    meteo['H2O'] = meteo['q']

    # calculate pressure levels
    meteo['p'] = utils.pressure_levels(ds_meteo['sp'].data, layers['hyai'], layers['hybi'])

    return meteo

def load_sim_props(data_dir: str, prefix: str, band: str):
    # load other relevant global atmospheric and surface properties from netCDF file
    sim_file = f'{prefix}_sim.nc'
    var_list = ['aerosol_optical_thickness', 'cirrus_optical_thickness', 'surface_albedo']
    ds_sim = xr.open_dataset(osp.join(data_dir, sim_file), group=band)
    props = {var: ds_sim[var].data for var in var_list}
    props['log_surface_albedo'] = np.log(props['surface_albedo'])
    return props


def load_trace_gas_total_columns(data_dir: str, prefix: str, gas_names: List[str]):
    # extract trace gas concentrations from netCDF file
    gas_data = {}
    for gas_name in gas_names:
        gas_file = f'{prefix}_{gas_name}.nc'
        ds = xr.open_dataset(osp.join(data_dir, gas_file))

        gas_name = gas_name.upper()
        column_total = utils.total_column_amount(ds[gas_name].data, ds['sp'].data, ds['hyai'].data, ds['hybi'].data)
        column_avg = utils.column_averaged_amount(ds[gas_name].data, ds['sp'].data, ds['hyai'].data, ds['hybi'].data)

        gas_data.update({
            f'{gas_name}_column_total': column_total,
            f'{gas_name}_column_avg': column_avg,
            f'log_{gas_name}_column_total': np.log(column_total),
        })

    return gas_data

def load_trace_gas_profiles(data_dir: str, prefix: str, gas_names: str):
    # extract trace gas vertical profiles from netCDF file
    gas_data = {}
    for gas_name in gas_names:
        gas_file = f'{prefix}_{gas_name}.nc'
        ds = xr.open_dataset(osp.join(data_dir, gas_file))
        gas_name = gas_name.upper()
        gas_data[gas_name] = ds[gas_name].data # shape: (nobs, nlevels)

    return gas_data

def load_cirrus(data_dir: str, prefix: str):
    # extract cirrus cloud properties from netCDF file
    c_file = f'{prefix}_cirrus.nc'
    ds_c = xr.open_dataset(osp.join(data_dir, c_file))

    var_list = ['tau_ref', 'top_height', 'geom_thickness']
    cirrus = {}
    for var in var_list:
        cirrus[f'cirrus_{var}'] = ds_c[var].data
    return cirrus

def load_aerosols(data_dir: str, prefix: str):
    # extract aerosol properties from netCDF file
    aer_file = f'{prefix}_inconsistent_aerosol.nc'
    ds_aer = xr.open_dataset(osp.join(data_dir, aer_file))

    aer = {
        'aer_tau_ref': ds_aer['tau_ref'].data,
    }
    var_list_modes = ['reff', 'veff', 'aer_col_relative', 'm_real', 'm_imag']
    for m in ds_aer['nmode'].data:
        ds_mode = ds_aer.sel(nmode=m)
        for var in var_list_modes:
            if var.startswith('m_'):
                for b in ds_mode['nband'].data:
                    ds_band = ds_mode.sel(nband=b)
                    aer[f'{var}_mode_{m}_band_{b}'] = ds_band[var].data
            else:
                if var.startswith('aer'):
                    aer[f'{var}_mode_{m}'] = ds_mode[var].data
                else:
                    aer[f'aer_{var}_mode_{m}'] = ds_mode[var].data
    return aer

def load_aerosol_profiles(data_dir: str, prefix: str):
    # extract aerosol properties from netCDF file
    aer_file = f'{prefix}_inconsistent_aerosol.nc'
    ds_aer = xr.open_dataset(osp.join(data_dir, aer_file))

    aer = {}
    aer['aer_p'] = ds_aer['press_lay'].data
    for m in ds_aer['nmode'].data:
        ds_mode = ds_aer.sel(nmode=m)
        nrel = ds_mode['aer_numb_relative'].data
        aer[f'aer_numb_relative_mode_{m}'] = nrel

    return aer

def load_spectra(data_dir: str, prefix: str, band: str, 
                 postfix_transmission: str = 'transmission',
                 wavelength: bool = True):

    # load data from full-physics simulation
    ff_file = f'{prefix}_sim.nc'
    ff_spectra = xr.open_dataset(osp.join(data_dir, ff_file), group=band)

    spectra = {
        'irradiance': ff_spectra['irradiance'].data,
        'radiance': ff_spectra['radiance'].data,
        }
    
    if wavelength:
        spectra['wavelength'] = ff_spectra['wavelength'].data
    
    # data from non-scattering simulation (transmission only)
    tr_file = f'{prefix}_sim_{postfix_transmission}.nc'
    if osp.exists(osp.join(data_dir, tr_file)):
        tr_spectra = xr.open_dataset(osp.join(data_dir, tr_file), group=band)
        spectra['transmission'] = tr_spectra['radiance'].data
    
    return spectra
    
def load_wavelengths(data_dir: str, prefix: str, band: str):
    # extract wavelengths of spectra from netCDF file
    sim_file = f'{prefix}_sim.nc'
    wl = xr.open_dataset(osp.join(data_dir, sim_file), group=band)['wavelength'].data
    return wl # shape: (nwave)


class RTDataset(Dataset):
    """Synthetic measurements generated with LINTRAN forward model"""
    def __init__(self, 
                 data_dir: str,
                 prefix: str = 's5_caseE',
                 pixel_features: Union[str, List[str]] = 'all',
                 profile_features: Union[str, List[str]] = 'all',
                 target_quantity: str = 'radiance',
                 predict_difference: bool = False,
                 pixel_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 profile_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 target_transform: utils.RTMTransform = utils.NumpyToTensor(),
                 seed: int = 0,
                 train: bool = True,
                 test_size: float = 0.2,
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

        # collect all relevant data
        self.pixel_data = self._collect_pixel_data(data_dir, prefix)
        self.profile_data = self._collect_profile_data(data_dir, prefix)
        self.targets, self.proxy = self._collect_spectra(data_dir, prefix)
        self.wavelengths = self._load_wavelengths(data_dir, prefix)
        self.indices = np.arange(self.targets.shape[0])

        # identify dimensions
        self.n_levels = self.profile_data.shape[1]
        self.n_wavelengths = self.targets.shape[1]
        self.n_pixel_features = len(self.pixel_features)
        self.n_profile_features = len(self.profile_features)

        # fix random seed for reproducibility
        self.seed = seed

        # define train/test split
        self.train = train
        self.train_indices, self.test_indices = self.get_data_split(test_size)
        self.indices = self.train_indices if self.train else self.test_indices

        self.pixel_data = self.pixel_data[self.indices]
        self.profile_data = self.profile_data[self.indices]
        self.targets = self.targets[self.indices]
        self.proxy = self.proxy[self.indices] if self.proxy is not None else None

        if self.predict_difference:
            self.targets = self.targets - self.proxy


    def set_transforms(self, pixel_transform=None, profile_transform=None, target_transform=None):
        self.pixel_transform = pixel_transform
        self.profile_transform = profile_transform
        self.target_transform = target_transform

    def add_transforms(self, pixel_transform=None, profile_transform=None, target_transform=None):
        if pixel_transform:
            self.pixel_transform = utils.RTMCompose([self.pixel_transform, pixel_transform]) \
                if self.pixel_transform else pixel_transform
        if profile_transform:
            self.profile_transform = utils.RTMCompose([self.profile_transform, profile_transform]) \
                if self.profile_transform else profile_transform
        if target_transform:
            self.target_transform = utils.RTMCompose([self.target_transform, target_transform]) \
                if self.target_transform else target_transform
            
    def get_transforms(self):
        return self.pixel_transform, self.profile_transform, self.target_transform
    
    def set_transforms(self, pixel_transform=None, profile_transform=None, target_transform=None):
        if pixel_transform is not None:
            self.pixel_transform = pixel_transform
        if profile_transform is not None:
            self.profile_transform = profile_transform
        if target_transform is not None:
            self.target_transform = target_transform
            
    def add_normalization(self, indices: List[int] = None, per_wavelength: bool = True):
        self.add_transforms(*self.normalization(indices, per_wavelength))
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):

        x_pixels = self.pixel_data[idx]
        x_profiles = self.profile_data[idx]
        y = self.targets[idx]

        if self.pixel_transform:
            x_pixels = self.pixel_transform(x_pixels)
        if self.profile_transform:
            x_profiles = self.profile_transform(x_profiles)
        if self.target_transform:
            y = self.target_transform(y)

        data_dict = {
            'pixel_inputs': x_pixels,
            'profile_inputs': x_profiles,
            'targets': y,
        }

        if self.proxy is not None:
            proxy = self.proxy[idx]
            if self.target_transform: # and not self.predict_difference:
                # if difference is predicted, leave proxy as is so that 
                # we can reconstruct the final outputs with it
                proxy = self.target_transform(proxy)
            data_dict['proxy_targets'] = proxy

        return data_dict
    
    def _collect_pixel_data(self, data_dir: str, prefix: str):
        # load pixel netcdf file
        pixel_file = f'{prefix}_pixel_data.nc'
        ds = xr.open_dataset(osp.join(data_dir, pixel_file))

        # extract relevant features
        if self.pixel_features == 'all':
            self.pixel_features = list(ds.data_vars.keys())
        pixel_data = np.stack([ds[feature].data for feature in self.pixel_features], axis=-1)
        
        return pixel_data # shape: (nobs, nvars)
    
    def _collect_profile_data(self, data_dir: str, prefix: str):
        # load profile netcdf file
        profile_file = f'{prefix}_profile_data.nc'
        ds = xr.open_dataset(osp.join(data_dir, profile_file))

        # extract relevant features
        if self.profile_features == 'all':
            self.profile_features = list(ds.data_vars.keys())
        profile_data = np.stack([ds[feature].data for feature in self.profile_features], axis=-1)
        
        return profile_data # shape: (nobs, nlevels, nvars)
    
    def _collect_spectra(self, data_dir: str, prefix: str):
        # load spectra netcdf file
        spectra_file = f'{prefix}_spectra.nc'
        ds = xr.open_dataset(osp.join(data_dir, spectra_file))

        # extract full physics spectra
        full_physics = ds[self.target_quantity].data # shape (nobs, nwave)

        # extract transmission spectra, if available
        if 'transmission' in ds.data_vars:
            transmission = ds['transmission'].data # shape (nobs, nwave)
        else:
            transmission = None
        
        return full_physics, transmission
    
    def _load_wavelengths(self, data_dir: str, prefix: str):
        # load wavelengths from spectra netcdf file
        spectra_file = f'{prefix}_spectra.nc'
        ds = xr.open_dataset(osp.join(data_dir, spectra_file))
        wavelengths = ds['wavelength'].data
        return wavelengths
        
    
    def normalization(self, indices: List[int] = None, per_wavelength: bool = True):
        if indices is None: indices = range(len(self))
        pixel_stats = self.get_pixel_statistics(indices)
        profile_stats = self.get_profile_statistics(indices)
        target_stats = self.get_target_statistics(indices, per_wavelength)
        pixel_norm = utils.normalization(pixel_stats)
        profile_norm = utils.normalization(profile_stats)
        target_norm = utils.normalization(target_stats)

        return pixel_norm, profile_norm, target_norm
    
    def get_pixel_statistics(self, indices: List[int]):
        if self.pixel_transform:
            data = self.pixel_transform(self.pixel_data[indices])
        else:
            data = self.pixel_data[indices]

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
    
    def get_profile_statistics(self, indices: List[int]):
        if self.profile_transform:
            data = self.profile_transform(self.profile_data[indices])
        else:
            data = self.profile_data[indices]

        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # compute stats
        stats = {
            'mean': data.mean(axis=(0, 1)),
            'std': data.std(axis=(0, 1)),
            'min': data.min(axis=(0, 1)),
            'max': data.max(axis=(0, 1)),
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
        profile_data = self.profile_data.reshape(-1, self.n_levels * self.n_profile_features)
        self.pixel_data = np.concatenate([self.pixel_data, profile_data], axis=-1)

        self.n_pixel_features = self.n_pixel_features + self.n_profile_features * self.n_levels
    
    def __getitem__(self, idx: int):

        x = self.pixel_data[idx]
        y = self.targets[idx]

        if self.pixel_transform:
            x = self.pixel_transform(x)
        if self.target_transform:
            y = self.target_transform(y)


        data_dict = {
            'inputs': x,
            'targets': y,
        }

        if self.proxy is not None:
            proxy = self.proxy[idx]
            if self.target_transform:
                proxy = self.target_transform(proxy)
            data_dict['proxy_targets'] = proxy

        return data_dict


    
