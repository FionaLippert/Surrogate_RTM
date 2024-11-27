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
from scipy.interpolate import interp1d

from rtm import utils

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
