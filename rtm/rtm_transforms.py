import numpy as np
import torch
from typing import Union, List, Dict, Callable, Optional
import os

class RTMTransform(torch.nn.Module):
    """
    Base class for RTM transforms. All transforms should inherit from 
    this class and implement the forward, inverse, and derivative methods.
    This allows for easy composition of transforms and their inverses, as well as
    computation of derivatives for use in optimization and sensitivity analysis.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply some transformation to data"""
        raise NotImplementedError
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the inverse transformation"""
        raise NotImplementedError
    
    def derivative(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the derivative of the transformation"""
        raise NotImplementedError
    
class RTMCompose(RTMTransform):
    """
    Compose multiple RTM transforms into a single transform. 
    The transforms are applied in the order they are provided.
    Args:
        transforms: List of RTMTransform instances to be composed.
    """
    def __init__(self, transforms: List[RTMTransform]):
        super().__init__()

        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        for transform in reversed(self.transforms):
            data = transform.inverse(data)
        return data
    
    def derivative(self, data, **kwargs):
        # iteratively compose the derivatives of the
        # transformations using the chain rule
        dtdx = 1
        dx = data
        for transform in self.transforms:
            dx = transform.derivative(dx, **kwargs)
            dtdx = dtdx * dx
        return dtdx
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.RTMCompose',
            'transforms': [t.__dict_repr__() for t in self.transforms]
        }
    
    def __repr__(self):
        return f'RTMCompose({self.transforms})'
    
    
class RTMTransformCollection(torch.nn.Module):
    """
    A collection of RTM transforms, each associated with a name.
    This allows for managing multiple transforms that may be applied to different
    parts of the data (e.g., pixel, profile, target, proxy).
    """
    def __init__(self, transforms: Dict[str, RTMTransform], **kwargs):
        super().__init__()

        self.names = list(transforms.keys())

        for name, t in transforms.items():
            setattr(self, name, t)

    def add_transforms(self, transforms: Dict[str, RTMTransform]):
        # add new transforms to existing transforms
        for name, t_new in transforms.items():
            if name in self.names:
                t_old = getattr(self, name, None)
                if isinstance(t_old, RTMTransform):
                    if isinstance(t_new, RTMTransform):
                        # compose new with existing transform
                        t_new = RTMCompose([t_old, t_new])
                    else:
                        # throw warning and keep old transform
                        print(f'{t_new} is not an RTMTransform... Ignore.')
                        t_new = t_old
                # set new transform
                setattr(self, name, t_new)
            else:
                raise ValueError(f'no transform registered under name {name}')
            
    def add_transform_to_all(self, transform: RTMTransform):
        for name in self.names:
            t_old = getattr(self, name, None)
            if isinstance(t_old, RTMTransform):
                t_new = RTMCompose([t_old, transform])
            else:
                t_new = transform
            setattr(self, name, t_new)
            
    def set_transforms(self, transforms: Dict[str, RTMTransform]):
        # overwrite existing transforms
        for name, t_new in transforms.items():
            if name in self.names:
                setattr(self, name, t_new)
    
    def get_transforms(self):
        # return [getattr(self, name) for name in self.names]
        return {name: getattr(self, name) for name in self.names}
    
    def get(self, name, default=None):
        return getattr(self, name, default)
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.RTMTransformCollection',
            'transforms': {name: getattr(self, name).__dict_repr__() for name in self.names}
        }
    
class NullTransform(RTMTransform):
    """
    A dummy transform that does nothing. This can be used as a 
    placeholder or default transform.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # do nothing to the data
        return data
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # do nothing to the data
        return data
    
    def derivative(self, data, **kwargs):
        return torch.ones_like(data)
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.NullTransform'}
    
    def __repr__(self):
        return 'NullTransform()'

class DoubleToFloat(RTMTransform):
    """
    A transform that converts double precision tensors to single precision (float32).
    """
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # convert to float32 tensor
        return data.to(torch.float)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(torch.double)
    
    def derivative(self, data, **kwargs):
        return torch.ones_like(data).to(torch.float)
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.DoubleToFloat'}
    
    def __repr__(self):
        return 'DoubleToFloat()'
    
class Normalize(RTMTransform):
    """
    Normalization transform that standardizes data to have zero mean and unit variance.

    Args:
        mean: The mean to use for normalization.
        std: The standard deviation to use for normalization.
        shape: The shape of the mean and std tensors. If not provided, defaults to 1.
    """
    def __init__(self, mean: Optional[torch.Tensor] = None, 
                 std: Optional[torch.Tensor] = None, 
                 shape: Optional[Union[int, List[int]]] = 1):
        super().__init__()

        if mean is None:
            mean = torch.zeros(shape)
        if std is None:
            std = torch.ones(shape)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    @property
    def shape(self) -> Union[int, List[int]]:
        return self.mean.shape

    def set_stats(self, mean: torch.Tensor = None, std: torch.Tensor = None):
        if mean is not None:
            self.mean = mean

        if std is not None:
            self.std = std

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # standardize to zero mean and unit variance
        # normalized = (data - self.mean.to(data.device)) / self.std.to(data.device)

        # get minimum value of data type to avoid division by zero
        min_value = torch.finfo(data.dtype).tiny

        normalized = (data - self.mean) / torch.clamp(self.std, min=min_value)
        return normalized
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # reverse standardization
        denormalized = data * self.std + self.mean

        return denormalized
    
    def derivative(self, data, **kwargs):

        # get minimum value of data type to avoid division by zero
        min_value = torch.finfo(data.dtype).tiny
        return torch.ones_like(data) / torch.clamp(self.std, min=min_value)
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.Normalize',
            'shape': self.shape
        }
    
    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std})'

class LogTransform(RTMTransform):
    """
    A transform that applies the natural logarithm to the data.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        # get minimum value of data type to avoid log(0)
        min_value = torch.finfo(data.dtype).tiny

        # apply log transform
        result = torch.log(torch.clamp(data, min=min_value))

        return result
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:

        # clamp data to avoid overflow or underflow in exp
        result = torch.exp(torch.clamp(data, min=-50, max=50))
        return result
    
    def derivative(self, data, **kwargs):

        # get minimum value of data type to avoid division by zero
        min_value = torch.finfo(data.dtype).tiny

        # now compute derivative
        der = 1 / torch.clamp(data, min=min_value)

        return der
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.LogTransform'}
    
    def __repr__(self):
        return 'LogTransform()'
    
class LogTransformSelectedFeatures(RTMTransform):
    """
    A transform that applies the natural logarithm to selected features of the data.
    
    Args:
        feature_indices: Indices of the features to which the log transform will be applied.
        n_features: Total number of features in the data.
        feature_dim: The dimension of the features in the data. Default is 1.
    """
    def __init__(self, 
                 feature_indices: List[int], 
                 n_features: int,
                 feature_dim: int = 1,
):
        super().__init__()
        self.feature_indices = feature_indices
        self.n_features = n_features
        self.feature_dim = feature_dim

        # assert that all feature indices are within bounds
        assert all(0 <= idx < n_features for idx in feature_indices), "Feature indices are out of bounds"

        # create a mask for the selected features
        mask = torch.zeros(n_features, dtype=torch.bool)
        mask[feature_indices] = True

        
        if feature_dim > 1:
            mask = mask.view(-1, 1).expand(-1, feature_dim)

        self.register_buffer('mask', mask)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # apply log transform to selected features

        # adjust dimensionality to allow for broadcasting
        new_shape = (-1, ) + (1,) * (data.dim() - self.feature_dim - 1)

        # get minimum value of data type to avoid log(0)
        min_value = torch.finfo(data.dtype).tiny

        # combine log-transformed data with original data
        mask = self.mask.view(new_shape).to(data.device)
        log_data = torch.log(torch.clamp(data, min=min_value)) * mask
        original_data = data * (torch.logical_not(mask))
        data = log_data + original_data

        return data

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # apply inverse log transform to selected features

        # adjust dimensionality to allow for broadcasting
        new_shape = (-1, ) + (1,) * (data.dim() - self.feature_dim - 1)

        # combine exp-transformed data with original data
        mask = self.mask.view(new_shape).to(data.device)
        exp_data = torch.exp(data) * mask
        original_data = data * (torch.logical_not(mask))
        data = exp_data + original_data

        return data
    
    
    def derivative(self, data, **kwargs):
        # derivative of log transform is 1/data
        # but only for the selected features
        # everywhere else it is 1.0
        new_shape = (-1, ) + (1,) * (data.dim() - self.feature_dim - 1)
        mask = self.mask.view(new_shape).to(data.device)

        # minimum value of data type to avoid division by zero
        min_value = torch.finfo(data.dtype).tiny

        # combine derivatives
        derivative = 1 / torch.clamp(data, min=min_value)
        derivative = derivative * mask + (1 - mask)

        return derivative
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.LogTransformSelectedFeatures',
                'feature_indices': self.feature_indices,
                'n_features': self.n_features,
                'feature_dim': self.feature_dim}
    
    def __repr__(self):
        return f'LogTransformSelectedFeatures(' + \
               f'feature_indices={self.feature_indices}, ' + \
               f'n_features={self.n_features}, ' + \
               f'feature_dim={self.feature_dim})'

class PowerTransform(RTMTransform):
    """
    A transform that raises the data to a specified power.

    Args:
        exponent: The power to which the data will be raised.
    """
    def __init__(self, exponent: float):
        super().__init__()

        self.register_buffer('exponent', exponent)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.pow(data, self.exponent)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return torch.pow(data, 1/self.exponent)
    
    def derivative(self, data, **kwargs):
        return self.exponent * torch.pow(data, self.exponent - 1)
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.PowerTransform',
            'exponent': self.exponent
        }
    
    def __repr__(self):
        return f'PowerTransform(exponent={self.exponent})'
    
    
    
def normalization(stats: Dict[str, Union[np.ndarray, torch.Tensor]], 
                  dtype: torch.dtype = torch.float64) -> Callable:
    """
    Creates a normalization transform based on the provided statistics.

    Args:
        stats: A dictionary containing 'mean' and 'std' for normalization.
        dtype: The desired data type for the mean and std tensors.
    Returns:
        A Normalize transform initialized with the provided mean and std.
    """
    mean = stats['mean']
    std = stats['std']

    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean.copy())
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std.copy())

    if isinstance(mean, (float, np.floating)):
        mean = torch.tensor(mean)
    if isinstance(std, (float, np.floating)):
        std = torch.tensor(std)

    mean = mean.to(dtype=dtype)
    std = std.to(dtype=dtype)

    # if no variation, replace with 1 to avoid division by zero
    std = torch.where(std==0, torch.ones_like(std), std)

    transform = Normalize(mean, std)

    return transform


def load_transforms(dir: str, types=['pixel', 'profile', 'target', 'proxy']):
    """
    Loads previously saved transforms from disk. If a transform file does not exist, 
    it returns None for that transform.
    """
    transforms = []
    for t in types:
        fname = os.path.join(dir, f'{t}_transform.pt')
        if os.path.isfile(fname):
            transf = torch.load(fname, weights_only=False)
            transforms.append(transf)
        else:
            transforms.append(None)
    return transforms

def save_transforms(transforms, dir: str, types=['pixel', 'profile', 'target', 'proxy']):
    """
    Saves transforms to disk. Each transform is saved with a filename corresponding to its type.
    """
    assert len(transforms) == len(types)
    for i, t in enumerate(types):
        fname = os.path.join(dir, f'{t}_transform.pt')
        torch.save(transforms[i], fname)
    

def numpy_to_tensor(data: np.ndarray, dtype=torch.float64) -> torch.Tensor:
    """Convert numpy array to Pytorch tensor"""
    return torch.from_numpy(data).to(dtype)

def tensor_to_numpy(data: torch.Tensor) -> np.ndarray:
    """Convert Pytorch tensor to numpy array"""
    return data.detach().cpu().numpy()
    
