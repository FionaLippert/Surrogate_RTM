import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ChainedScheduler
from hydra.utils import instantiate

from typing import Callable, Union, List, Dict, Tuple, Optional
from typing_extensions import TypeAlias

from rtm.constants import constants

from torch.types import Storage
MAP_LOCATION: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, dict[str, str]]
]

##################################################
# S5 specific functions
##################################################

def s5_band_to_wl_range(band: str) -> Tuple[int, int]:
    """Get wavelength range for a given band."""
    if band in ['NIR', '3']:
        wl_range = (745, 773)
    elif band in ['NIR1', '3a']:
        wl_range = (745, 755)
    elif band in ['NIR2', '3b']:
        wl_range = (755, 773)
    elif band in ['SWIR1', '4']:
        wl_range = (1590, 1675)
    elif band in ['SWIR3', '5']:
        wl_range = (2305, 2385)
    else:
        raise ValueError(f'Band {band} not recognized')
    return wl_range

def s5_band_to_iwin(band: str) -> int:
    """Get Fortran band ID for a given band."""
    if band in ['NIR', '3']:
        band_id = 1
    elif band in ['SWIR1', '4']:
        band_id = 2
    elif band in ['SWIR3', '5']:
        band_id = 3
    else:
        raise ValueError(f'Band {band} not supported by NN forward model')
    
    return band_id

def s5_band_to_absorbers(band: str) -> List[str]:
    """Get list of absorbers for a given band."""
    if band in ['NIR', 'NIR1', 'NIR2', '3', '3a', '3b']:
        absorbers = ['o2']
    elif band in ['SWIR1', '4']:
        absorbers = ['h2o', 'co2', 'ch4']
    elif band in ['SWIR3', '5']:
        absorbers = ['h2o', 'co', 'ch4']
    else:
        raise ValueError(f'Band {band} not recognized')
    return absorbers

def gas_to_hitran_index(gas: str) -> int:
    """Get HITRAN index for a given gas."""
    gas2hitran = {
        'h2o': 1,
        'co2': 2,
        'o2': 7,
        'co': 5,
        'ch4': 6,
    }
    return gas2hitran[gas.lower()]

def hitran_index_to_gas(index: int) -> str:
    """Get gas name for a given HITRAN index."""
    hitran2gas = {
        1: 'h2o',
        2: 'co2',
        5: 'co',
        6: 'ch4',
        7: 'o2',
    }
    return hitran2gas[index]

##################################################
# radiative transfer model specific functions
##################################################

def azimuth_difference(azimuth_1: np.ndarray, azimuth_2: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute difference between two azimuth angles, 
    accounting for circularity.
    """
    diff = np.abs(azimuth_1 - azimuth_2)
    diff = np.where(diff > 180, 360 - diff, diff)
    return diff

def total_column_amount(column_concentrations: np.ndarray,
                        plev: np.ndarray = None,
                        sp: np.ndarray = None,
                        a: np.ndarray = None,
                        b: np.ndarray = None,
                        ) -> np.ndarray:
    """
    Calculate the total column amount of a trace gas given its column 
    concentrations and pressure levels.

    Note that either plev, or sp, a, b must be provided. 
    If plev is not provided, it will be calculated from surface pressure (sp) 
    and coefficients a and b.

    Args:
        column_concentrations: Trace gas column concentrations [kg/kg] (shape: nobs, nlevels)
        plev: Pressure levels [hPa] (shape: nobs, nlevels+1)
        sp: Surface pressure [hPa] (shape: nobs)
        a: Coefficient a for pressure level calculation (shape: nlevels+1)
        b: Coefficient b for pressure level calculation (shape: nlevels+1)
    Returns:
        Total column amount of the trace gas [kg/m^2] (shape: nobs)
    """

    if plev is None:
        # calculate pressure levels from surface pressure and coefficients a and b
        plev = pressure_half_levels(sp, a, b)

    # calculate number of air molecules per layer [#/m^2]
    vc_air = air_column(plev) # total air column above each pressure level
    vc_air = vc_air[:, 1:] - vc_air[:, :-1] # above half-level 0, air pressure is zero

    # calculate total column amount of trace gas
    vc_gas = column_concentrations * vc_air
    total_gas = np.sum(vc_gas, axis=-1)

    return total_gas

def total_water(specific_humidity: np.ndarray,
                plev: np.ndarray = None,
                sp: np.ndarray = None,
                a: np.ndarray = None,
                b: np.ndarray = None,
                ) -> np.ndarray:
    """
    Calculate the total precipitable water content given 
    specific humidity and pressure levels.

    Note that either plev, or sp, a, b must be provided.
    If plev is not provided, it will be calculated from surface pressure (sp) 
    and coefficients a and b.

    Args:
        specific_humidity: Specific humidity [kg/kg] (shape: nobs, nlevels)
        plev: Pressure levels [hPa] (shape: nobs, nlevels+1)
        sp: Surface pressure [hPa] (shape: nobs)
        a: Coefficient a for pressure level calculation (shape: nlevels+1)
        b: Coefficient b for pressure level calculation (shape: nlevels+1)
    Returns:
        Total precipitable water content [kg/m^2] (shape: nobs)
    """

    if plev is None:
        # calculate pressure levels from surface pressure and coefficients a and b
        plev = pressure_half_levels(sp, a, b)

    # calculate total precipitable water content [kg/m^2]
    delta_p = plev[:, 1:] - plev[:, :-1] # pressure level thickness
    g = constants["gravity"]
    tpw = g * np.sum(specific_humidity * delta_p, axis=-1)

    return tpw

def column_averaged_water(specific_humidity: np.ndarray,
                          plev: np.ndarray = None,
                          sp: np.ndarray = None,
                          a: np.ndarray = None,
                          b: np.ndarray = None,
                          ) -> np.ndarray:
    """
    Calculate the column-averaged specific humidity given 
    specific humidity and pressure levels.

    Note that either plev, or sp, a, b must be provided.
    If plev is not provided, it will be calculated from surface pressure (sp) 
    and coefficients a and b.

    Args:
        specific_humidity: Specific humidity [kg/kg] (shape: nobs, nlevels)
        plev: Pressure levels [hPa] (shape: nobs, nlevels+1)
        sp: Surface pressure [hPa] (shape: nobs)
        a: Coefficient a for pressure level calculation (shape: nlevels+1)
        b: Coefficient b for pressure level calculation (shape: nlevels+1)
    Returns:
        Column-averaged specific humidity [kg/kg] (shape: nobs)
    """

    if plev is None:
        # calculate pressure levels from surface pressure and coefficients a and b
        plev = pressure_half_levels(sp, a, b)

    # calculate column-averaged specific humidity [kg/kg]
    delta_p = plev[:, 1:] - plev[:, :-1] # pressure level thickness
    q_avg = np.sum(specific_humidity * delta_p, axis=-1) / np.sum(delta_p, axis=-1)

    return q_avg

def specific_humididty_to_mmr(specific_humidity: np.ndarray) -> np.ndarray:
    """Convert specific humidity to mass mixing ratio."""
    mmr = specific_humidity / (1 - specific_humidity)
    return mmr

def column_averaged_mole_fraction(column_mmr: np.ndarray, 
                                  gas: str,
                                  plev: np.ndarray = None,
                                  sp: np.ndarray = None,
                                  a: np.ndarray = None,
                                  b: np.ndarray = None,
                                  ) -> np.ndarray:
    """
    Calculate the column-averaged mole fraction given 
    mass mixing ratio and pressure levels.

    Note that either plev, or sp, a, b must be provided.
    If plev is not provided, it will be calculated from surface pressure (sp) 
    and coefficients a and b.

    Args:
        column_mmr: Mass mixing ratio in kg/kg (shape: nobs, nlevels)
        gas: Name of the trace gas
        plev: Pressure levels [hPa] (shape: nobs, nlevels+1)
        sp: Surface pressure [hPa] (shape: nobs)
        a: Coefficient a for pressure level calculation (shape: nlevels+1)
        b: Coefficient b for pressure level calculation (shape: nlevels+1)
    Returns:
        Column-averaged mole fraction in mol/mol (shape: nobs)
    """

    if plev is None:
        # calculate pressure levels from surface pressure and coefficients a and b
        plev = pressure_half_levels(sp, a, b)

   # calculate pressure level differences
    delta_p = plev[:, 1:] - plev[:, :-1]

    # calculate total column mass of trace gas
    tc_gas = np.sum(column_mmr * delta_p / constants["gravity"], axis=-1) # in kg/m^2
    tc_gas = tc_gas / constants["mass"][gas] # in mol/m^2

    # calculate total column amount of air
    tc_air = np.sum(delta_p / constants["gravity"], axis=-1) # in kg/m^2
    tc_air = tc_air / constants["mass"]["dry_air"] # in mol/m^2

    # calculate column-averaged mole fraction
    mole_fraction = tc_gas / tc_air # in mol/mol

    return mole_fraction

def column_averaged_amount(column_concentrations: np.ndarray,
                           plev: np.ndarray = None,
                           sp: np.ndarray = None,
                           a: np.ndarray = None,
                           b: np.ndarray = None,
                           ) -> np.ndarray:
    """
    Calculate the column-averaged amount of a trace gas given its concentrations and pressure levels.

    Note that either plev, or sp, a, b must be provided.
    If plev is not provided, it will be calculated from surface pressure (sp) 
    and coefficients a and b.

    Args:
        column_concentrations: Concentrations of the trace gas (shape: nobs, nlevels)
        plev: Pressure levels [hPa] (shape: nobs, nlevels+1)
        sp: Surface pressure [hPa] (shape: nobs)
        a: Coefficient a for pressure level calculation (shape: nlevels+1)
        b: Coefficient b for pressure level calculation (shape: nlevels+1)
    Returns:
        Column-averaged amount of the trace gas (shape: nobs)
    """

    if plev is None:
        # calculate pressure levels from surface pressure and coefficients a and b
        plev = pressure_half_levels(sp, a, b)
                                    
    # calculate pressure level differences
    delta_p = plev[:, 1:] - plev[:, :-1]

    # calculate column-averaged amount of trace gas
    vc_gas = column_concentrations * delta_p
    gas_avg = np.sum(vc_gas, axis=-1) / np.sum(delta_p, axis=-1)

    return gas_avg

def pressure_half_levels(sp: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate pressure half-levels from surface pressure and coefficients a and b.

    Args:
        sp: Surface pressure in hPa (shape: nobs)
        a: Coefficient a (shape: nlayers+1)
        b: Coefficient b (shape: nlayers+1)
    Returns:
        Pressure half-levels in hPa (shape: nobs, nlayers+1)
    """
    plev = np.expand_dims(a, axis=0) + np.expand_dims(b, axis=0) * np.expand_dims(sp, axis=-1)

    return plev # shape: (nobs, nlayers+1)

def pressure_layer_midpoints(sp: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate pressure layer midpoints from surface pressure and coefficients a and b.

    Args:
        sp: Surface pressure in hPa (shape: nobs)
        a: Coefficient a (shape: nlayers+1)
        b: Coefficient b (shape: nlayers+1)
    Returns:
        Pressure layer midpoints in hPa (shape: nobs, nlayers)
    """ 
    plev = np.expand_dims(a, axis=0) + np.expand_dims(b, axis=0) * np.expand_dims(sp, axis=-1)

    # from half-levels to full layers
    p_mid = pressure_half_levels_to_midpoints(plev)
    return p_mid # shape: (nobs, nlayers)

def pressure_half_levels_to_midpoints(plev: np.ndarray) -> np.ndarray:
    """Convert pressure half-levels to layer midpoints"""
    p_mid = 0.5 * (plev[:, :-1] + plev[:, 1:])
    return p_mid

def air_column(p: np.ndarray) -> np.ndarray:
    """
    Calculate the total air column above each pressure level.

    Args:
        p: Pressure levels in hPa (shape: nobs, nlevels)
    Returns:
        Total air column above each pressure level in mol/m^2 (shape: nobs, nlevels)
    """
    n_a = constants["avogadro"]
    m_air = constants["mass"]["dry_air"]
    g = constants["gravity"]
    vc_air = p * n_a / (m_air * g)

    return vc_air

def global_to_profile(data: np.ndarray, nlevels: int) -> np.ndarray:
    """
    Convert global data to profile data by repeating vertically.

    Args:
        data: Global data (shape: nobs)
        nlevels: Number of vertical levels
    Returns:
        Profile data (shape: nobs, nlevels)
    """
    profile = np.expand_dims(data, axis=-1)
    profile = np.tile(profile, (1, nlevels))

    return profile

def mmr_to_ppm(mmr: np.ndarray, gas: str) -> np.ndarray:
    """
    Convert mass mixing ratio to parts per million (ppm).

    Args:
        mmr: Mass mixing ratio (shape: nobs)
        gas: Name of the gas
    Returns:
        Parts per million (ppm) (shape: nobs)
    """
    m_air = constants["mass"]["dry_air"]
    m_gas = constants["mass"][gas]
    concentration = mmr * m_air / m_gas
    concentration = 1e6 * concentration

    return concentration

def dry_mmr_to_wet_mmr(dry_xh2o: np.ndarray) -> np.ndarray:
    """
    Convert dry mass mixing ratio of H2O to moist mass mixing ratio 
    (i.e. specific humidity).

    Args:
        dry_xh2o: Dry mass mixing ratio of H2O (shape: nobs)
    Returns:
        Specific humidity (shape: nobs)
    """
    # convert dry MMR of H2O to moist MMR (i.e. specific humidity)
    specific_humidity = dry_xh2o / (1 + dry_xh2o)
    return specific_humidity



##################################################
# Evaluation metrics
##################################################

def mape(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Mean absolute percentage error [%]"""
    return 100 * np.mean(np.abs((outputs - targets) / targets), **kwargs)

def mape_continuum(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Mean absolute percentage error [%] w.r.t continuum"""
    return 100 * np.mean(np.abs((outputs - targets) / targets.max(1, keepdims=True)), **kwargs)

def mse(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Mean-squared error"""
    return np.mean((outputs - targets) ** 2, **kwargs)

def mae(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Mean absolute error"""
    return np.mean(np.abs(outputs - targets), **kwargs)

def fvu(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Fraction of variance unexplained [%]"""
    return 100 * mse(targets, outputs, **kwargs) / np.var(targets, **kwargs)

def rmse(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Root-mean-squared error"""
    return np.sqrt(mse(outputs, targets, **kwargs))

def rmse_continuum(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Root-mean-squared error w.r.t continuum"""
    mse_continuum = np.mean(((outputs - targets) / targets.max(1, keepdims=True)) ** 2, **kwargs)
    rmse_continuum = 100 * np.sqrt(mse_continuum)
    return rmse_continuum

def nrmse(outputs: np.ndarray, 
          targets: np.ndarray, 
          z: np.ndarray = None,
          **kwargs) -> np.ndarray:
    """Spectral normalized root-mean-squared error [%] (Servera et al., 2021)"""
    if z is None:
        # separate value for each wavelength
        max_target = targets.max(**kwargs)
        min_target = targets.min(**kwargs)
        z = max_target - min_target
    return 100 * rmse(outputs, targets, **kwargs) / z

def smape(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Symmetric mean absolute percentage error [%]"""
    fracs = np.abs(outputs - targets) / (np.abs(outputs) + np.abs(targets))
    fracs = np.where(np.isnan(fracs), np.zeros_like(fracs), fracs)
    return 100 * np.mean(fracs, **kwargs)

def bias(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Bias, i.e. mean error"""
    return np.mean(outputs - targets, **kwargs)

def relative_bias(outputs: np.ndarray, 
                  targets: np.ndarray, 
                  z: np.ndarray = None, 
                  **kwargs) -> np.ndarray:
    """Relative bias, i.e. mean percentage error [%]"""
    if z is None:
        z = targets
    return 100 * np.mean((outputs - targets) / z, **kwargs)

def relative_rmse(outputs: np.ndarray, 
                  targets: np.ndarray, 
                  z: np.ndarray = None,
                  **kwargs) -> np.ndarray:
    """Relative root-mean-squared error w.r.t some reference z"""
    if z is None:
        z = targets
    mse = np.mean(((outputs - targets) / z) ** 2, **kwargs)
    rmse = 100 * np.sqrt(mse)
    return rmse

def r2_score(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Coefficient of determination [%]"""
    ss_res = np.sum((targets - outputs) ** 2, **kwargs)
    ss_tot = np.sum((targets - np.mean(targets, **kwargs)) ** 2, **kwargs)
    return 100 * (1 - ss_res / ss_tot)




##################################################
# Model training
##################################################

def init_weights(model, mode='fan_out'):
    """Initialize model weights with Kaiming method for relu activations"""
    if type(model) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(model.weight, nonlinearity='relu', mode=mode)
        # torch.nn.init.kaiming_uniform_(model.weight, a=0, mode='fan_out', nonlinearity='relu')
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)

def freeze_weights(model):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_weights(model):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def get_device(accelerator: str = 'gpu', **kwargs):
    """Check for GPU availability"""
    if accelerator in ['cuda', 'gpu'] and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_lr_scale(module: torch.nn.Module):
    """Get learning rate scaling factor for NN module"""
    if hasattr(module, 'lr_scale'):
        print(module.lr_scale)
        return module.lr_scale
    else:
        # if lr_scale is not set for module, use default learning rate
        return 1.0
    

##################################################
# Learning rate scheduling
##################################################

class LRDampenedCosine:
    """
    Dampened cosine learning rate scheduler with exponential decay.

    Args:
        lr_min: Minimum learning rate.
        lr_max: Maximum learning rate.
        T_max: Number of epochs for one cosine cycle.
        gamma: Exponential decay factor.
        gamma_interval: Number of epochs between each decay step.
    """
    def __init__(self, lr_min=0.0, lr_max=1.0, T_max=10, gamma=0.95, gamma_interval=1):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.T_max = T_max
        self.gamma = gamma
        self.gamma_interval = gamma_interval

    def __call__(self, epoch):
        cosine = np.cos(epoch / self.T_max * np.pi)
        scaling = self.gamma ** (epoch // self.gamma_interval)
        lr_out = self.lr_min + scaling * 0.5 * (self.lr_max - self.lr_min) * (1 + cosine)

        return lr_out
    
class CustomChainedScheduler(ChainedScheduler):
    """
    ChainedScheduler that allows for partially defined schedulers to be passed in.
    This is useful for schedulers that require an optimizer to be passed in,
    but the optimizer is not yet defined at the time of scheduler creation.
    """
    def __init__(self, optimizer, schedulers, **kwargs):

        # initialize partially defined lr schedulers
        schedulers = [s_partial(optimizer) for s_partial in schedulers]

        super().__init__(schedulers, optimizer=optimizer, **kwargs)


##################################################
# Model loading
##################################################

def load_model_from_hydra_config(hydra_config: Dict) -> torch.nn.Module:
    """Load model from hydra config"""
    model = instantiate(hydra_config['model'])
    transforms = instantiate(hydra_config['transforms'])
    model.transforms = transforms

    return model

def load_model_from_checkpoint(checkpoint: Dict, map_location: MAP_LOCATION = None) -> torch.nn.Module:
    """Load model from checkpoint"""
    model = load_model_from_hydra_config(checkpoint['hydra_config'])
    model.load_model_state(checkpoint, map_location=map_location)

    return model