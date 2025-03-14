import os
import os.path as osp
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Dict, Tuple

from rtm import utils

def save_netcdf_file(data: xr.Dataset, save_path: str):
    if os.path.isfile(save_path):
        # remove existing file
        os.remove(save_path)
    data.to_netcdf(save_path, format='NETCDF4')
    data.close()


def combine_chunks(data_dir: str, prefix: str, n_chunks: int, combined_dir: str = 'combined'):

    if combined_dir is not None:
        out_dir = osp.join(data_dir, combined_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = data_dir
    
    # find all pixel data files and combine them
    pixel_files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('_pixel_data.nc')]
    pixel_files.sort()
    pixel_files = pixel_files[:n_chunks]
    pixel_data = combine_netcdf_files(data_dir, pixel_files)
    save_netcdf_file(pixel_data, osp.join(out_dir, f'{prefix}_combined_pixel_data.nc'))

    # find all profile data files and combine them
    profile_files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('_profile_data.nc')]
    profile_files.sort()
    profile_files = profile_files[:n_chunks]
    profile_data = combine_netcdf_files(data_dir, profile_files)
    save_netcdf_file(profile_data, osp.join(out_dir, f'{prefix}_combined_profile_data.nc'))

    # find all spectral data files and combine them
    spectral_files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('_spectra.nc')]
    spectral_files.sort()
    spectral_files = spectral_files[:n_chunks]
    spectral_data = combine_netcdf_files(data_dir, spectral_files)
    save_netcdf_file(spectral_data, osp.join(out_dir, f'{prefix}_combined_spectra.nc'))

    # find all coordinate files and combine them
    coord_files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('_coords.nc')]
    coord_files.sort()
    coord_files = coord_files[:n_chunks]
    coord_data = combine_netcdf_files(data_dir, coord_files)
    save_netcdf_file(coord_data, osp.join(out_dir, f'{prefix}_combined_coords.nc'))


def combine_netcdf_files(data_dir: str, file_list: List[str]):
    # combine chunked netCDF files into a single file
    for i, f in enumerate(file_list):
        ds = xr.open_dataset(osp.join(data_dir, f))
        if i == 0:
            combined = ds
        else:
            # adjust pixelID to avoid duplicates
            ds['pixelID'] = ds['pixelID'] + combined['pixelID'].max() + 1
            combined = xr.concat([combined, ds], dim='pixelID')
        ds.close()

    return combined

def prepare_data(target_dir: str,
                 data_dir: str,
                 prefix: str,
                 wl_range: Tuple[float, float],
                 gas_names: List[str] = ['co2', 'ch4'],
                 instrument: str = 'instr2_noise0',
                 prefix_proxy: str = None,
                ):
    
    os.makedirs(target_dir, exist_ok=True)

    # load spectral data
    spectral_data = load_spectra(data_dir, prefix, wl_range, 
                                 instrument=instrument, 
                                 prefix_proxy=prefix_proxy)

    # load coordinates
    coords = load_coords(data_dir, prefix, 
                         instrument=instrument)

    # load pixel data
    pixel_data = load_pixel_data(data_dir, prefix, wl_range,
                                 gas_names=gas_names, 
                                 instrument=instrument)
    if 'proxy' in spectral_data:
        # add proxy radiances as additional features
        pixel_data['proxy'] = spectral_data['proxy']

    # load profile data
    profile_data = load_profile_data(data_dir, prefix, gas_names)

    # save data to disk
    spectra_to_netcdf(spectral_data, target_dir, prefix)
    coords_to_netcdf(coords, target_dir, prefix)
    pixel_data_to_netcdf(pixel_data, target_dir, prefix)
    profile_data_to_netcdf(profile_data, target_dir, prefix)



def pixel_data_to_netcdf(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format and write to disk
    for k, v in data.items():
        if len(v.shape) > 1:
            v = v.reshape(v.shape[0], -1)
            data[k] = {'dims': ('pixelID', f'dim_{k}'), 'data': v}
        else:
            data[k] = {'dims': ('pixelID'), 'data': v}
    data['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    data = xr.Dataset.from_dict(data)

    save_netcdf_file(data, osp.join(target_dir, f'{prefix}_pixel_data.nc'))

def pixel_data_to_csv(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # save pixel data to csv file
    df_pixels = pd.DataFrame(data)
    df_pixels['pixelID'] = df_pixels.index

    save_path = osp.join(target_dir, f'{prefix}_pixel_data.csv')
    if os.path.isfile(save_path):
        # remove existing file
        os.remove(save_path)
    df_pixels.to_csv(save_path, index=False)

def coords_to_netcdf(coords: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format and write to disk
    for k, v in coords.items():
        coords[k] = {'dims': ('pixelID'), 'data': v}
    coords['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    coords = xr.Dataset.from_dict(coords)

    save_netcdf_file(coords, osp.join(target_dir, f'{prefix}_coords.nc'))

def coords_to_csv(coords: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # save coordinates to csv file
    df_coords = pd.DataFrame(coords)
    df_coords['pixelID'] = df_coords.index

    save_path = osp.join(target_dir, f'{prefix}_coords.csv')
    if os.path.isfile(save_path):
        # remove existing file
        os.remove(save_path)
    df_coords.to_csv(save_path, index=False)


def profile_data_to_netcdf(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format
    for k, v in data.items():
        data[k] = {'dims': ('pixelID', 'layerID'), 'data': v}
    data['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(v.shape[0])}
    data['layerID'] = {'dims': ('layerID'), 'data': np.arange(v.shape[1])}
    data = xr.Dataset.from_dict(data)

    save_netcdf_file(data, osp.join(target_dir, f'{prefix}_profile_data.nc'))

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
    save_path = osp.join(target_dir, f'{prefix}_profile_data.csv')
    if os.path.isfile(save_path):
        # remove existing file
        os.remove(save_path)
    df_profiles.to_csv(save_path, index=False)

def spectra_to_netcdf(data: Dict[str, np.ndarray], target_dir: str, prefix: str):
    # convert dict to netcdf format

    for k, v in data.items():
        if k == 'wavelength':
            data[k] = {'dims': ('wavelength'), 'data': v}
        else:
            data[k] = {'dims': ('pixelID', 'wavelength'), 'data': v}
            npix = v.shape[0]
    data['pixelID'] = {'dims': ('pixelID'), 'data': np.arange(npix)}
    data = xr.Dataset.from_dict(data)

    save_netcdf_file(data, osp.join(target_dir, f'{prefix}_spectra.nc'))

def spectra_to_csv(data: Dict[str, np.ndarray], target_dir: str, prefix: str, quantity: str):
    # save spectra (only for the given quantity) to csv file
    df_spectra = pd.DataFrame(data[quantity], columns=data['wavelength'])
    df_spectra['pixelID'] = df_spectra.index

    save_path = osp.join(target_dir, f'{prefix}_spectra.csv')
    if os.path.isfile(save_path):
        # remove existing file
        os.remove(save_path)
    df_spectra.to_csv(save_path, index=False)

def load_pixel_data(data_dir: str,
                      prefix: str,
                      wl_range: Tuple[float, float],
                      gas_names: List[str] = ['co2', 'ch4'],
                      instrument: str = 'instr2_noise0',
                     ):
    # load pixel/surface properties that do not come as vertical profiles
    data = {}
    data.update(load_geometry(data_dir, prefix, instrument))
    data.update(load_surface(data_dir, prefix, gas_names))
    data.update(load_state_vars(data_dir, prefix, wl_range, gases=gas_names))

    return data

def load_profile_data(data_dir: str,
                         prefix: str,
                         gas_names: List[str] = ['co2', 'ch4'],
                        ):
    # load vertical profiles of trace gases and meteo variables
    data = load_atmospheric_profiles(data_dir, prefix, gas_names)

    return data


def load_geometry(data_dir: str, prefix: str, instrument: str = 'instr2_noise0'):
    # extract orbit geometry from measurement NetCDF file
    fname = f'{prefix}_{instrument}.nc'
    var_list = [
        'SolarZenithAngle', 
        'ViewingZenithAngle', 
        'RelativeAzimuthAngle', 
        'ScatteringAngle',
        'Flight_Altitude',
        'Pixel_Altitude',
        ]
    ds = xr.open_dataset(osp.join(data_dir, fname))
    geom_dict = {var: ds[var].data.squeeze() for var in var_list}
    ds.close()

    vars = list(geom_dict.keys())
    for var in vars:
        if 'Angle' in var:
            data = geom_dict[var]
            geom_dict[f'cos_{var}'] = np.cos(np.deg2rad(data))
            geom_dict[f'sin_{var}'] = np.sin(np.deg2rad(data))

    geom_dict['air_mass_factor'] = (1 / geom_dict['cos_SolarZenithAngle']) + \
                                   (1 / geom_dict['cos_ViewingZenithAngle'])
    
    return geom_dict


def load_coords(data_dir: str, prefix: str, instrument: str = 'instr2_noise0'):
    fname = f'{prefix}_{instrument}.nc'
    # extract datetime, longitude and latitude from netCDF file
    var_list = ['UTC_Date', 'Fracday', 'Julday', 'Lon', 'Lat']
    ds = xr.open_dataset(osp.join(data_dir, fname))
    coords = {var: ds[var].data for var in var_list}
    ds.close()

    # TODO: convert UTC_Date (int) and Fracday (float) to datetime?

    return coords
        
def load_surface(data_dir: str, prefix: str, gas_names: List[str]):
    # extract surface properties and meteo data from netCDF file
    fname = f'{prefix}-aux.nc'
    var_list = ['surface_pressure_meteo', 
                'surface_elevation_meteo', 
                'surface_elevation_dem', 
                'surface_elevation_std',
                'land_fraction',
                'cloud_fraction',
                'cirrus_fraction',
                'tropopause_height',
                'cloud_top_height_asl']
    ds = xr.open_dataset(osp.join(data_dir, fname))
    surface_dict = {var: ds[var].data for var in var_list}

    if 'plev' in ds:
        kwargs = {'plev': ds['plev'].data}
    else:
        kwargs = {
            'sp': ds['surface_pressure_meteo'].data,
            'a': ds['hybrid_a'].data,
            'b': ds['hybrid_b'].data
        }

    # apply cube-root transformation to surface elevation
    surface_dict['cbrt_surface_elevation_meteo'] = np.cbrt(surface_dict['surface_elevation_meteo'])
    surface_dict['cbrt_surface_elevation_dem'] = np.cbrt(surface_dict['surface_elevation_dem'])

    # calculate total water vapor
    tpw = utils.total_water(ds['specific_humidity'].data, **kwargs)
    surface_dict['h2o_column_total'] = tpw
    surface_dict['log_h2o_column_total'] = np.log(np.maximum(tpw, 1e-12))
    surface_dict['h2o_column_avg'] = utils.column_averaged_water(ds['specific_humidity'].data, **kwargs)

    MMR_h2o = utils.specific_humididty_to_mmr(ds['specific_humidity'].data)
    surface_dict['xh2o'] = utils.column_averaged_mole_fraction(MMR_h2o, 'h2o', **kwargs)

    # calculate total column amounts for trace gases
    for gas in gas_names:
        gas_data = ds[f'MMR_{gas}'].data
        column_total = utils.total_column_amount(gas_data, **kwargs)
        column_avg = utils.column_averaged_amount(gas_data, **kwargs)
        avg_mole_fraction = utils.column_averaged_mole_fraction(gas_data, gas, **kwargs)

        surface_dict[f'{gas}_column_total'] = column_total
        surface_dict[f'{gas}_column_avg'] = column_avg
        surface_dict[f'log_{gas}_column_total'] = np.log(np.maximum(column_total, 1e-12))
        surface_dict[f'x{gas}'] = avg_mole_fraction

    ds.close()

    return surface_dict

def load_atmospheric_profiles(data_dir: str, prefix: str, gas_names: List[str]):
    # extract verical meteo profiles from netCDF file
    fname = f'{prefix}-aux.nc'
    var_list = ['specific_humidity', 'temperature']
    ds = xr.open_dataset(osp.join(data_dir, fname))
    atm_dict = {var: ds[var].data for var in var_list}
    layers = {var: ds[var].data for var in ['hybrid_a', 'hybrid_b']}

    # apply log-transform and cube-root transformation to specific humidity
    atm_dict['log_specific_humidity'] = np.log(np.maximum(atm_dict['specific_humidity'], 1e-12))
    atm_dict['cbrt_specific_humidity'] = np.cbrt(atm_dict['specific_humidity'])
    atm_dict['MMR_h2o'] = utils.specific_humididty_to_mmr(atm_dict['specific_humidity'])

    # calculate pressure at layer midpoints
    if 'plev' in ds:
        atm_dict['pressure'] = utils.pressure_half_levels_to_midpoints(ds['plev'].data)
    else:
        atm_dict['pressure'] = utils.pressure_layer_midpoints(
            ds['surface_pressure_meteo'].data, 
            layers['hybrid_a'], 
            layers['hybrid_b']
        )
    atm_dict['log_pressure'] = np.log(np.maximum(atm_dict['pressure'], 1e-12))

    # extract MMR profiles for trace gases
    for gas in gas_names:
        atm_dict[f'MMR_{gas}'] = ds[f'MMR_{gas}'].data

    ds.close()

    return atm_dict

def load_state_vars(data_dir: str, 
                    prefix: str, 
                    wl_range: Tuple[float, float], 
                    gases: List[str] = ['co2', 'ch4'], 
                    nmodes: int = 2):
    
    # load ground truth state variables (aerosol and surface characteristics, etc) from netCDF file
    fname = f'{prefix}-truth.nc'
    ds = xr.open_dataset(osp.join(data_dir, fname))

    props = {}

    # extract total column denisty for trace gases
    for gas in gases:
        hitran_idx = utils.gas_to_hitran_index(gas)
        props[f'vmol_{gas}'] = ds[f'Truth_Vmol'].data[:, hitran_idx]

    # find wavelengths in Wave_Opt that are within the specified range of the considered band
    wl_opt = ds['Wave_Opt'].data
    wl_opt_idx = np.where((wl_opt >= wl_range[0]) & (wl_opt <= wl_range[1]))[0]
    wl = ds['Wavelength'].data
    wl_idx = np.where((wl >= wl_range[0]) & (wl <= wl_range[1]))[0]
    wl_band = ds['Band_Win'].data
    wl_band_idx = np.where((wl_band >= wl_range[0]) & (wl_band <= wl_range[1]))[0]

    # extract total aerosol optical thickness at wavelengths within band
    props['AOT'] = ds['Truth_AOT'].data[:, wl_opt_idx].squeeze() # total

    # extract aerosol absorption optical thickness at wavelengths within band
    props['AAOT'] = ds['Truth_AAOT'].data[:, wl_opt_idx].squeeze() # total

    # extract aerosol single scattering albedo at wavelengths within band
    props['SSA'] = ds['Truth_SSA'].data[:, wl_opt_idx].squeeze()

    # extract mode-specific aerosol parameters
    for m in range(nmodes):
        # aerosol optical thickness
        props[f'AOT_mode{m+1}'] = ds['Truth_Mode_AOT'].data[:, wl_opt_idx, m].squeeze() # at wavelengths within band
        props[f'AOT_550_mode{m+1}'] = ds[f'Truth_Mode{m+1}_TAU550'].data # at 550nm

        # aerosol absorption optical thickness
        props[f'AAOT_mode{m+1}'] = ds['Truth_Mode_AAOT'].data[:, wl_opt_idx, m].squeeze() # at wavelengths within band

        # effective radius and variance
        props[f'reff_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Aerosol_Pars'].data[:, 0]
        props[f'veff_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Aerosol_Pars'].data[:, 1]


        # real and imaginary refractive index
        props[f'RRI_550_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Aerosol_Pars'].data[:, 2] # at 550nm
        props[f'RRI_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Mr'].data[:, wl_idx].squeeze() # at central wavelength
        props[f'IRI_550_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Aerosol_Pars'].data[:, 3] # at 550nm
        props[f'IRI_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Mi'].data[:, wl_idx].squeeze() # at central wavelength

        # aerosol column density
        props[f'N_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Aerosol_Pars'].data[:, 4]

        # spherical fraction for fine and coarse mode
        props[f'f_sphere_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Aerosol_Pars'].data[:, 5]

        # species fractions
        nspecies = ds[f'Truth_Mode{m+1}_Refrcoef'].shape[1]
        for i in range(nspecies):
            props[f'vfrac_mode{m+1}_species{i+1}'] = ds[f'Truth_Mode{m+1}_Refrcoef'].data[:, i]

        # center of aerosol height distribution
        props[f'z_aer_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Xalt'].data[:, 0]
        props[f'w_aer_mode{m+1}'] = ds[f'Truth_Mode{m+1}_Xalt'].data[:, 1]

    # extract Ross-Li BRDF parameters
    props['k_geo'] = ds['Truth_Xbdrf_Geo'].data[:, 4]
    props['k_vol'] = ds['Truth_Xbdrf_Geo'].data[:, 5]

    # extract surface albedo
    props['albedo'] = ds['Truth_Xbdrf_Wave_Av'].data[:, wl_idx].squeeze() # at central wavelength
    for i in range(ds['Truth_Xbdrf_Wave_Par'].shape[1]):
        # get albedo coefficient of polynomial fit
        props[f'albedo_coef{i}'] = ds['Truth_Xbdrf_Wave_Par'].data[:, i, wl_band_idx].squeeze()

    # extract Maigan-Breon XBPDF parameters
    props['B_pol'] = ds['Truth_Xbpdf'].data[:, 0]
    props['v_pol'] = ds['Truth_Xbpdf'].data[:, 3]

    ds.close()

    # apply log-transform where applicable
    vars = props.keys()
    log_props = {}
    for var in vars:
        if np.all(props[var] >= 0):
            log_props[f'log_{var}'] = np.log(np.maximum(props[var], 1e-12))
    props.update(log_props)

    return props


def load_spectra(data_dir: str, prefix: str, wl_range: str, 
                 prefix_proxy: str = None,
                 wavelength: bool = True,
                 instrument: str = 'instr2_noise0',):

    # load data from full-physics simulation
    fname = f'{prefix}_{instrument}.nc'

    ds = xr.open_dataset(osp.join(data_dir, fname))

    # find indices of wavelengths in specified range
    wl = ds['Wavelength'].data
    wl_idx = np.where((wl >= wl_range[0]) & (wl <= wl_range[1]))[0]

    spectra = {}

    # get radiance spectra for specified wavelength range
    spectra['radiance'] = ds['Radiance'].data[:, 0, wl_idx]
    
    if wavelength:
        spectra['wavelength'] = ds['Wavelength'].data[wl_idx]
    
    if prefix_proxy is not None:
        # add data from non-scattering, transmission, or single-scattering simulation
        tr_fname = f'{prefix_proxy}_{instrument}.nc'
        if osp.exists(osp.join(data_dir, tr_fname)):
            tr_spectra = xr.open_dataset(osp.join(data_dir, tr_fname))
            # get prior spectra for specified wavelength range
            spectra['proxy'] = tr_spectra['Radiance'].data[:, 0, wl_idx]
        else:
            print(f'No proxy spectra found for {prefix_proxy}')

    ds.close()
        
    return spectra
    
def load_wavelengths(data_dir: str, prefix: str, 
                     wl_range: Tuple[float, float], 
                     instrument: str = 'instr2_noise0'):
    # extract wavelengths of spectra from netCDF file
    fname = f'{prefix}_{instrument}.nc'
    ds = xr.open_dataset(osp.join(data_dir, fname))
    wl = ds['wavelength'].data

    # select wavelength range
    wl = wl[wl >= wl_range[0] and wl <= wl_range[1]]

    ds.close()

    return wl # shape: (nwave)
