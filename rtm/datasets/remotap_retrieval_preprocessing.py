import xarray as xr
import netCDF4 as nc
import numpy as np
import os
from os import path as osp
import warnings
from rtm import utils


def save_netcdf_file(
        data: xr.Dataset, 
        save_path: str):
    """
    Save xarray Dataset to NetCDF file, removing existing file if it exists.
    """
    if os.path.isfile(save_path):
        # remove existing file
        os.remove(save_path)
    data.to_netcdf(save_path, format='NETCDF4')
    data.close()

def prepare_data(
        target_dir: str,
        data_dir: str,
        prefix: str = 'remotap_retrieval',
        iwin: int = 1,
        npix: int = 1000,
        max_chi2: float = 10000.0):
    """
    Prepare data for training/finetuning surrogate RTM from retrieval results,
    and save processed data to disk in NetCDF format. 

    Args:
        target_dir: Directory to save processed data to.
        data_dir: Directory containing retrieval results.
        prefix: Prefix for RemoTAP output files.
        iwin: Retrieval window index.
        npix: Number of pixels to process.
        max_chi2: Maximum chi-squared value for filtering valid retrievals.
    """
    
    os.makedirs(target_dir, exist_ok=True)

    # loop over batches of 1000 pixels, save resulting data and concatenate at the end
    spectral_data = []
    pixel_data = []
    jacobian_data = []
    for i in range(0, npix, 1000):
        # load retrieval data
        ret_data = load_retrieval_data(data_dir, iwin=iwin, ipix_min=i, ipix_max=i+1000)

        # filter data based on chi2
        valid = filter_data(ret_data, max_chi2=max_chi2)

        # load spectral data
        spectral_data.append(prepare_spectra(ret_data, valid))

        # load pixel data
        pixel_data.append(prepare_pixel_data(ret_data, valid))

        # load jacobians
        jacobian_data.append(prepare_jacobians(ret_data, valid))

        if 'proxy' in spectral_data[-1]:
            # add proxy radiances as additional features
            pixel_data[-1]['proxy'] = spectral_data[-1]['proxy']

    # concatenate data from all batches
    spectral_data = xr.concat(spectral_data, dim='pixelID')
    pixel_data = xr.concat(pixel_data, dim='pixelID')
    jacobian_data = xr.concat(jacobian_data, dim='pixelID')
            

    # save data to disk (and apply filtering)
    save_netcdf_file(spectral_data, osp.join(target_dir, f'{prefix}_spectra.nc'))
    save_netcdf_file(pixel_data, osp.join(target_dir, f'{prefix}_pixel_data.nc'))
    save_netcdf_file(jacobian_data, osp.join(target_dir, f'{prefix}_jacobians.nc'))


def load_retrieval_data(
        data_dir: str, 
        iwin: int, 
        ipix_min: int = 0, 
        ipix_max: int = 1000) -> xr.Dataset:
    """
    Load retrieval data from RemoTAP output files for a given window and pixel range.

    Args:
        data_dir: Directory containing retrieval results.
        iwin: Retrieval window index.
        ipix_min: Minimum pixel index to load (inclusive).
        ipix_max: Maximum pixel index to load (exclusive).
    Returns:
        ds_ret: xr.Dataset containing retrieval data for the specified pixel range.
    """
    ds_list = []
    for i in range(ipix_min, ipix_max):
        fp = f'{data_dir}/pix_{i+1:>05}_win_{iwin:02}.nc'
        if not osp.exists(fp):
            continue
        ds_list.append(xr.load_dataset(fp))

    ds_ret = xr.concat(ds_list, dim='npix')

    return ds_ret

def filter_data(
        ret_data: xr.Dataset, 
        max_chi2: float = 10000) -> np.ndarray:
    """
    Filter retrieval data based on chi-squared values.
    """
    # get valid retrieval iterations
    chi2 = ret_data.chi2.values # (npix, nitr)
    valid = (chi2 > 0) & (chi2 < max_chi2)

    print(f'Number of valid retrieval iterations: ' + \
          f'{np.sum(valid)} ({np.sum(valid)/valid.size*100:.1f}%)')

    return valid

def prepare_spectra(
        ret_data: xr.Dataset, 
        valid: np.ndarray) -> xr.Dataset:
    """
    Prepare target and proxy spectra for valid retrieval iterations 
    and combine into a single xarray Dataset.

    Args:
        ret_data: xarray Dataset containing retrieval data.
        valid: Boolean array indicating valid retrieval iterations.
    Returns:
        ds_spectra: xr.Dataset containing target and proxy spectra for valid iterations.
    """
    # get target spectra for valid iterations
    y_target = ret_data.radiance_ms.values # (npix, nwave, nitr)
    y_target = y_target.transpose(0, 2, 1) # (npix, nitr, nwave)
    y_target = y_target[valid] # (nvalid, nwave)

    # get proxy spectra for valid iterations
    y_proxy = ret_data.radiance_ns.values # (npix, nwave, nitr)
    y_proxy = y_proxy.transpose(0, 2, 1) # (npix, nitr, nwave)
    y_proxy = y_proxy[valid] # (nvalid, nwave)

    # get wavelengths
    if 'wavelength' in ret_data:
        wavelengths = ret_data.wavelength.values
    else:
        wavelengths = np.arange(y_target.shape[1])

    # combine wavelengths, target and proxy spectra into NetCDF file for data loading
    ds_spectra = xr.Dataset(
        {
            "wavelength": (["wavelength"], wavelengths),
            "radiance": (["pixelID", "wavelength"], y_target),
            "proxy": (["pixelID", "wavelength"], y_proxy),
        }
    )

    return ds_spectra

def get_feature_names(ret_data: xr.Dataset) -> list:
    """
    Get pixel feature names from retrieval data attributes, replacing HITRAN indices with gas names.
    """
    # get pixel feature names from attribute "n_in_varnames" (comma-separated string)
    feature_names = ret_data.attrs['n_in_varnames'].split(',') # list of strings

    # replace hitran indices with gas names for gas absorption features "vmol_X"
    for i, feature_name in enumerate(feature_names):
        if feature_name.startswith('vmol_'):
            try:
                hitran_index = int(feature_name.split('_')[1])
                gas_name = utils.hitran_index_to_gas(hitran_index)
                feature_names[i] = f'vmol_{gas_name}'
            except:
                # throw warning
                warnings.warn(f'Unknown HITRAN index in feature name: {feature_name}')
                pass

    return feature_names

def prepare_pixel_data(
        ret_data: xr.Dataset, 
        valid: np.ndarray) -> xr.Dataset:
    """
    Prepare pixel features for valid retrieval iterations and combine into a single xarray Dataset.
    """
    # get pixel features for valid iterations
    x_in = ret_data.nn_x_in.values # (npix, nfeatures, nitr)
    x_in = x_in.transpose(0, 2, 1) # (npix, nitr, nfeatures)
    x_in = x_in[valid] # (nvalid, nfeatures)

    # get pixel feature names from attribute "n_in_varnames" (comma-separated string)
    feature_names = get_feature_names(ret_data)

    # combine pixel features into NetCDF file for data loading
    # iterate over feature names and create separate variable for each feature
    ds_pixel_data = xr.Dataset()
    for i, feature_name in enumerate(feature_names):
        ds_pixel_data[feature_name] = (["pixelID"], x_in[:, i])

    return ds_pixel_data

def prepare_jacobians(
        ret_data: xr.Dataset, 
        valid: np.ndarray) -> xr.Dataset:
    """
    Prepare target and proxy jacobians for valid retrieval iterations and 
    combine into a single xarray Dataset.
    """
    # get target derivatives for valid iterations
    dy_target = ret_data.kmat.values # (npix, nfeatures, nwave, nitr)
    dy_target = dy_target.transpose(0, 3, 2, 1) # (npix, nitr, nwave, nfeatures)
    dy_target = dy_target[valid] # (nvalid, nwave, nfeatures)

    # get proxy derivatives for valid iterations
    dy_proxy = ret_data.kmat_ns.values # (npix, nfeatures, nwave, nitr)
    dy_proxy = dy_proxy.transpose(0, 3, 2, 1) # (npix, nitr, nwave, nfeatures)
    dy_proxy = dy_proxy[valid] # (nvalid, nwave, nfeatures)

    # get prior total gas columns for valid iterations
    vmol_prior = ret_data.vmol_prior.values # (npix, ngases, nitr)
    vmol_prior = vmol_prior.transpose(0, 2, 1) # (npix, nitr, ngases)
    vmol_prior = vmol_prior[valid] # (nvalid, ngases)

    # get pixel feature names from attribute "n_in_varnames" (comma-separated string)
    feature_names = get_feature_names(ret_data)

    # combine pixel jacobians into NetCDF file for data loading
    # iterate over feature names and create separate variable for each feature
    ds_jacobians = xr.Dataset()
    imol = 0
    for i, feature_name in enumerate(feature_names):
        ds_jacobians[f'{feature_name}_jac'] = (["pixelID", "wavelength"], dy_target[:, :, i])
        ds_jacobians[f'{feature_name}_proxy_jac'] = (["pixelID", "wavelength"], dy_proxy[:, :, i])

        if feature_name.startswith('vmol_'):
            # add prior total gas column as additional feature
            ds_jacobians[f'{feature_name}_prior'] = (["pixelID"], vmol_prior[:, imol])
            imol += 1

    return ds_jacobians