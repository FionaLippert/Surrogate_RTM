import numpy as np
import torch
from torchvision.transforms import Compose
from typing import Callable, Union, List, Dict, Tuple

from rtm.constants import constants

##################################################
# radiative transfer model specific functions
##################################################

def azimuth_difference(azimuth_1: np.ndarray, azimuth_2: np.ndarray) -> np.ndarray:
    diff = np.abs(azimuth_1 - azimuth_2)
    diff = np.where(diff > 180, 360 - diff, diff)
    return diff

def total_column_amount(column_concentrations: np.ndarray,
                        sp: np.ndarray,
                        a: np.ndarray,
                        b: np.ndarray
                        ) -> np.ndarray:
    # calculate number of air molecules per layer [#/m^2]
    p = pressure_half_levels(sp, a, b)
    vc_air = air_column(p) # total air column above each pressure level
    vc_air = vc_air[:, 1:] - vc_air[:, :-1] # above half-level 0, air pressure is zero

    # calculate total column amount of trace gas
    vc_gas = column_concentrations * vc_air
    total_gas = np.sum(vc_gas, axis=-1)

    return total_gas

def total_water(specific_humidity: np.ndarray,
                sp: np.ndarray,
                a: np.ndarray,
                b: np.ndarray
                ) -> np.ndarray:
    # calculate total precipitable water content [kg/m^2]
    p = pressure_half_levels(sp, a, b)
    delta_p = p[:, 1:] - p[:, :-1] # pressure level thickness
    g = constants["gravity"]
    tpw = g * np.sum(specific_humidity * delta_p, axis=-1)

    return tpw

def column_averaged_water(specific_humidity: np.ndarray,
                          sp: np.ndarray,
                          a: np.ndarray,
                          b: np.ndarray
                          ) -> np.ndarray:
    # calculate column-averaged specific humidity [kg/kg]
    p = pressure_half_levels(sp, a, b)
    delta_p = p[:, 1:] - p[:, :-1] # pressure level thickness
    q_avg = np.sum(specific_humidity * delta_p, axis=-1) / np.sum(delta_p, axis=-1)

    return q_avg

def column_averaged_amount(column_concentrations: np.ndarray,
                           sp: np.ndarray,
                           a: np.ndarray,
                           b: np.ndarray
                           ) -> np.ndarray:
    # calculate pressure level differences
    p = pressure_half_levels(sp, a, b)
    delta_p = p[:, 1:] - p[:, :-1]

    # calculate column-averaged amount of trace gas
    vc_gas = column_concentrations * delta_p
    gas_avg = 1e6 * np.sum(vc_gas, axis=-1) / np.sum(delta_p, axis=-1) # [ppm]

    return gas_avg

def pressure_half_levels(sp: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # calculate pressure levels from surface pressure and coefficients a and b
    # sp: surface pressure in hPa (shape: nobs)
    # a: coefficient a (shape: nlevels+1)
    # b: coefficient b (shape: nlevels+1)
    p_i = np.expand_dims(a, axis=0) + np.expand_dims(b, axis=0) * np.expand_dims(sp, axis=-1)

    return p_i # shape: (nobs, nlevels+1)

def pressure_levels(sp: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # calculate pressure levels from surface pressure and coefficients a and b
    # sp: surface pressure in hPa (shape: nobs)
    # a: coefficient a (shape: nlevels+1)
    # b: coefficient b (shape: nlevels+1)
    p_i = np.expand_dims(a, axis=0) + np.expand_dims(b, axis=0) * np.expand_dims(sp, axis=-1)

    # from half-levels to full levels
    p = 0.5 * (p_i[:, :-1] + p_i[:, 1:])
    return p # shape: (nobs, nlevels)

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


##################################################
# torch transforms
##################################################

class RTMTransform:
    def __init__(self):
        pass

    def __call__(self, data: Union[np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        # apply some transformation to data
        raise NotImplementedError
    
    def inverse(self, data: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        # reverse the transformation
        raise NotImplementedError
    
class RTMCompose(RTMTransform):
    def __init__(self, transforms: List[RTMTransform]):
        self.transforms = transforms

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def inverse(self, data: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        for transform in reversed(self.transforms):
            data = transform.inverse(data)
        return data
    
    def __repr__(self):
        return f'RTMCompose({self.transforms})'

class NumpyToTensor(RTMTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        # convert numpy array to tensor, inheriting the dtype
        return torch.from_numpy(data)
    
    def inverse(self, data: torch.Tensor) -> np.ndarray:
        # convert tensor back to numpy
        return data.detach().cpu().numpy()
    
    def __repr__(self):
        return 'NumpyToTensor()'
    
class DoubleToFloat(RTMTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # convert to float32 tensor
        return data.to(torch.float)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(torch.double)
    
    def __repr__(self):
        return 'DoubleToFloat()'
    
class Normalize(RTMTransform):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std


    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # standardize to zero mean and unit variance
        normalized = (data - self.mean.to(data.device)) / self.std.to(data.device)
        return normalized

    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # reverse standardization
        denormalized = data * self.std.to(data.device) + self.mean.to(data.device)
        return denormalized
    
    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std})'

class LogTransform(RTMTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return torch.exp(data)
    
    def __repr__(self):
        return 'LogTransform()'
    
class FourierTransform(RTMTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(data)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(data)
    
    def __repr__(self):
        return 'FourierTransform()'
    
class FTransform(RTMTransform):
    def __init__(self, f: Callable, f_inv: Callable):
        self.f = f
        self.f_inf = f_inv

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # apply custom function f to data
        return self.f(data)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # reverse custom function f
        return self.f_inv(data)
    
    def __repr__(self):
        return f'FTransform(f={self.f}, f_inv={self.f_inv})'
    
    
def normalization(stats: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Callable:
    mean = stats['mean']
    std = stats['std']

    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean.copy())
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std.copy())

    # if no variation, replace with 1 to avoid division by zero
    std = torch.where(std==0, torch.ones_like(std), std)

    transform = Normalize(mean, std)

    return transform



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
        torch.nn.init.kaiming_normal_(model.weight, nonlinearity='relu')

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
    