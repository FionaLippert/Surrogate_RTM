import os
import numpy as np
import torch
from torchvision.transforms import Compose
from torch.optim.lr_scheduler import ChainedScheduler
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

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
    # get wavelength range for a given band
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
    # get Fortran band ID for a given band
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
    gas2hitran = {
        'h2o': 1,
        'co2': 2,
        'o2': 7,
        'co': 5,
        'ch4': 6,
    }
    return gas2hitran[gas.lower()]

def hitran_index_to_gas(index: int) -> str:
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
    diff = np.abs(azimuth_1 - azimuth_2)
    diff = np.where(diff > 180, 360 - diff, diff)
    return diff

def total_column_amount(column_concentrations: np.ndarray,
                        plev: np.ndarray = None,
                        sp: np.ndarray = None,
                        a: np.ndarray = None,
                        b: np.ndarray = None,
                        ) -> np.ndarray:
    # Note: either plev, or sp, a, b must be provided

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
    # Note: either plev, or sp, a, b must be provided

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
    # Note: either plev, or sp, a, b must be provided

    if plev is None:
        # calculate pressure levels from surface pressure and coefficients a and b
        plev = pressure_half_levels(sp, a, b)

    # calculate column-averaged specific humidity [kg/kg]
    delta_p = plev[:, 1:] - plev[:, :-1] # pressure level thickness
    q_avg = np.sum(specific_humidity * delta_p, axis=-1) / np.sum(delta_p, axis=-1)

    return q_avg

def specific_humididty_to_mmr(specific_humidity: np.ndarray) -> np.ndarray:
    # convert specific humidity to mass mixing ratio
    mmr = specific_humidity / (1 - specific_humidity)
    return mmr

def column_averaged_mole_fraction(column_mmr: np.ndarray, 
                                  gas: str,
                                  plev: np.ndarray = None,
                                  sp: np.ndarray = None,
                                  a: np.ndarray = None,
                                  b: np.ndarray = None,
                                  ) -> np.ndarray:
    # calculate column-averaged mole fraction
    # column_mmr: mass mixing ratio in kg/kg (shape: nobs, nlevels)
    # return: column-averaged mole fraction in mol/mol (shape: nobs)

    # Note: either plev, or sp, a, b must be provided
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
    # Note: either plev, or sp, a, b must be provided

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
    # calculate pressure levels from surface pressure and coefficients a and b
    # sp: surface pressure in hPa (shape: nobs)
    # a: coefficient a (shape: nlayers+1)
    # b: coefficient b (shape: nlayers+1)
    plev = np.expand_dims(a, axis=0) + np.expand_dims(b, axis=0) * np.expand_dims(sp, axis=-1)

    return plev # shape: (nobs, nlayers+1)

def pressure_layer_midpoints(sp: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # calculate pressure layer midpoints from surface pressure and coefficients a and b
    # sp: surface pressure in hPa (shape: nobs)
    # a: coefficient a (shape: nlayers+1)
    # b: coefficient b (shape: nlayers+1)
    plev = np.expand_dims(a, axis=0) + np.expand_dims(b, axis=0) * np.expand_dims(sp, axis=-1)

    # from half-levels to full layers
    p_mid = pressure_half_levels_to_midpoints(plev)
    return p_mid # shape: (nobs, nlayers)

def pressure_half_levels_to_midpoints(plev: np.ndarray) -> np.ndarray:
    # convert pressure half-levels to layer midpoints
    p_mid = 0.5 * (plev[:, :-1] + plev[:, 1:])
    return p_mid

def air_column(p: np.ndarray) -> np.ndarray:
    # calculate air column above each pressure level
    # p: pressure levels in hPa (shape: nobs, nlevels)
    n_a = constants["avogadro"]
    m_air = constants["mass"]["dry_air"]
    g = constants["gravity"]
    vc_air = p * n_a / (m_air * g)

    return vc_air

def global_to_profile(data: np.ndarray, nlevels: int) -> np.ndarray:
    # convert global data to profile data by repeating vertically
    # data: global data (shape: nobs)
    # nlevels: number of vertical levels
    profile = np.expand_dims(data, axis=-1)
    profile = np.tile(profile, (1, nlevels))

    return profile

def mmr_to_ppm(mmr: np.ndarray, gas: str) -> np.ndarray:
    # convert mass mixing ratio to parts per million (ppm)
    m_air = constants["mass"]["dry_air"]
    m_gas = constants["mass"][gas]
    concentration = mmr * m_air / m_gas
    concentration = 1e6 * concentration

    return concentration

def dry_mmr_to_wet_mmr(dry_xh2o: np.ndarray) -> np.ndarray:
    # convert dry MMR of H2O to moist MMR (i.e. specific humidity)
    specific_humidity = dry_xh2o / (1 + dry_xh2o)
    return specific_humidity


# ##################################################
# # torch transforms
# ##################################################

# class RTMTransform:
#     def __init__(self):
#         pass

#     def __call__(self, data: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
#         # apply some transformation to data
#         raise NotImplementedError
    
#     def inverse(self, data: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
#         # reverse the transformation
#         raise NotImplementedError
    
# class RTMCompose(RTMTransform):
#     def __init__(self, transforms: List[RTMTransform]):
#         self.transforms = transforms

#     def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
#         for transform in self.transforms:
#             data = transform(data)
#         return data
    
#     def inverse(self, data: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
#         for transform in reversed(self.transforms):
#             data = transform.inverse(data)
#         return data
    
#     def __repr__(self):
#         return f'RTMCompose({self.transforms})'

# class NumpyToTensor(RTMTransform):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, data: np.ndarray) -> torch.Tensor:
#         # convert numpy array to tensor, inheriting the dtype
#         return torch.from_numpy(data)
    
#     def inverse(self, data: torch.Tensor) -> np.ndarray:
#         # convert tensor back to numpy
#         return data.detach().cpu().numpy()
    
#     def __repr__(self):
#         return 'NumpyToTensor()'
    
# class DoubleToFloat(RTMTransform):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, data: torch.Tensor) -> torch.Tensor:
#         # convert to float32 tensor
#         return data.to(torch.float)
    
#     def inverse(self, data: torch.Tensor) -> torch.Tensor:
#         return data.to(torch.double)
    
#     def __repr__(self):
#         return 'DoubleToFloat()'
    
# class Normalize(RTMTransform):
#     def __init__(self, mean: torch.Tensor, std: torch.Tensor):
#         super().__init__()
#         self.mean = mean
#         self.std = std


#     def __call__(self, data: torch.Tensor) -> torch.Tensor:
#         # standardize to zero mean and unit variance
#         normalized = (data - self.mean.to(data.device)) / self.std.to(data.device)
#         return normalized

    
#     def inverse(self, data: torch.Tensor) -> torch.Tensor:
#         # reverse standardization
#         denormalized = data * self.std.to(data.device) + self.mean.to(data.device)
#         return denormalized
    
#     def __repr__(self):
#         return f'Normalize(mean={self.mean}, std={self.std})'

# class LogTransform(RTMTransform):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, data: torch.Tensor) -> torch.Tensor:
#         return torch.log(data)
    
#     def inverse(self, data: torch.Tensor) -> torch.Tensor:
#         return torch.exp(data)
    
#     def __repr__(self):
#         return 'LogTransform()'
    
# class FourierTransform(RTMTransform):
#     def __init__(self):
#         super().__init__()

#     def __call__(self, data: torch.Tensor) -> torch.Tensor:
#         return torch.fft.fft(data)
    
#     def inverse(self, data: torch.Tensor) -> torch.Tensor:
#         return torch.fft.ifft(data)
    
#     def __repr__(self):
#         return 'FourierTransform()'
    
# class FTransform(RTMTransform):
#     def __init__(self, f: Callable, f_inv: Callable):
#         self.f = f
#         self.f_inf = f_inv

#     def __call__(self, data: torch.Tensor) -> torch.Tensor:
#         # apply custom function f to data
#         return self.f(data)
    
#     def inverse(self, data: torch.Tensor) -> torch.Tensor:
#         # reverse custom function f
#         return self.f_inv(data)
    
#     def __repr__(self):
#         return f'FTransform(f={self.f}, f_inv={self.f_inv})'
    
    
# def normalization(stats: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Callable:
#     mean = stats['mean']
#     std = stats['std']

#     if isinstance(mean, np.ndarray):
#         mean = torch.from_numpy(mean.copy())
#     if isinstance(std, np.ndarray):
#         std = torch.from_numpy(std.copy())

#     if isinstance(mean, (float, np.floating)):
#         mean = torch.tensor(mean)
#     if isinstance(std, (float, np.floating)):
#         std = torch.tensor(std)

#     # if no variation, replace with 1 to avoid division by zero
#     std = torch.where(std==0, torch.ones_like(std), std)

#     transform = Normalize(mean, std)

#     return transform


# def load_transforms(dir: str, types=['pixel', 'profile', 'target', 'proxy']):
#     transforms = []
#     for t in types:
#         fname = os.path.join(dir, f'{t}_transform.pt')
#         if os.path.isfile(fname):
#             transf = torch.load(fname)
#             transforms.append(transf)
#         else:
#             transforms.append(None)
#     return transforms

# def save_transforms(transforms, dir: str, types=['pixel', 'profile', 'target', 'proxy']):
#     assert len(transforms) == len(types)
#     for i, t in enumerate(types):
#         fname = os.path.join(dir, f'{t}_transform.pt')
#         torch.save(transforms[i], fname)
            



##################################################
# Evaluation metrics
##################################################

def mape(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # mean absolute percentage error [%]
    return 100 * np.mean(np.abs((outputs - targets) / targets), **kwargs)

def mape_continuum(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # mean absolute percentage error [%] w.r.t continuum
    return 100 * np.mean(np.abs((outputs - targets) / targets.max(1, keepdims=True)), **kwargs)

def mse(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # mean-squared error
    return np.mean((outputs - targets) ** 2, **kwargs)

def mae(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # mean absolute error
    return np.mean(np.abs(outputs - targets), **kwargs) 

def fvu(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # fraction of variance unexplained [%]
    return 100 * mse(targets, outputs, **kwargs) / np.var(targets, **kwargs)

def rmse(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # root-mean-squared error
    return np.sqrt(mse(outputs, targets, **kwargs))

def rmse_continuum(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # root-mean-squared error w.r.t continuum
    mse_continuum = np.mean(((outputs - targets) / targets.max(1, keepdims=True)) ** 2, **kwargs)
    rmse_continuum = 100 * np.sqrt(mse_continuum)
    return rmse_continuum

def nrmse(outputs: np.ndarray, 
          targets: np.ndarray, 
          z: np.ndarray = None,
          **kwargs) -> np.ndarray:
    # spectral normalized root-mean-squared error [%] (Servera et al., 2021)
    if z is None:
        # separate value for each wavelength
        max_target = targets.max(**kwargs)
        min_target = targets.min(**kwargs)
        z = max_target - min_target
    return 100 * rmse(outputs, targets, **kwargs) / z

def smape(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # symmetric mean absolute percentage error [%]
    fracs = np.abs(outputs - targets) / (np.abs(outputs) + np.abs(targets))
    fracs = np.where(np.isnan(fracs), np.zeros_like(fracs), fracs)
    return 100 * np.mean(fracs, **kwargs)

def bias(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # bias, i.e. mean error
    return np.mean(outputs - targets, **kwargs)

def relative_bias(outputs: np.ndarray, 
                  targets: np.ndarray, 
                  z: np.ndarray = None, 
                  **kwargs) -> np.ndarray:
    # relative bias, i.e. mean percentage error [%]
    if z is None:
        z = targets
    return 100 * np.mean((outputs - targets) / z, **kwargs)

def relative_rmse(outputs: np.ndarray, 
                  targets: np.ndarray, 
                  z: np.ndarray = None,
                  **kwargs) -> np.ndarray:
    # root-mean-squared error w.r.t some reference z
    if z is None:
        z = targets
    mse = np.mean(((outputs - targets) / z) ** 2, **kwargs)
    rmse = 100 * np.sqrt(mse)
    return rmse

def r2_score(outputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    # coefficient of determination [%]
    ss_res = np.sum((targets - outputs) ** 2, **kwargs)
    ss_tot = np.sum((targets - np.mean(targets, **kwargs)) ** 2, **kwargs)
    return 100 * (1 - ss_res / ss_tot)




##################################################
# Model training
##################################################

def init_weights(model):
    """Initialize model weights with Kaiming method for relu activations"""
    if type(model) == torch.nn.Linear:
        # torch.nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(model.weight, a=0, mode='fan_out', nonlinearity='relu')
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
    
class test_lr_func:
    def __init__(self):
        pass
    
    def __call__(self, epoch):
        return 10
    
class CustomChainedScheduler(ChainedScheduler):
    def __init__(self, optimizer, schedulers, **kwargs):

        # initialize partially defined lr schedulers
        schedulers = [s_partial(optimizer) for s_partial in schedulers]

        super().__init__(schedulers, optimizer=optimizer, **kwargs)


##################################################
# Model loading
##################################################

def load_model_from_hydra_config(hydra_config: Dict) -> torch.nn.Module:
    # load model from hydra config
    model = instantiate(hydra_config['model'])
    transforms = instantiate(hydra_config['transforms'])
    model.transforms = transforms

    return model

def load_model_from_checkpoint(checkpoint: Dict, map_location: MAP_LOCATION = None) -> torch.nn.Module:
    # load model from checkpoint
    model = load_model_from_hydra_config(checkpoint['hydra_config'])
    model.load_model_state(checkpoint, map_location=map_location)

    return model