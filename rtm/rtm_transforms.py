import numpy as np
import torch
from typing import Union, List, Dict, Callable, Optional
import os

class RTMTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        # apply some transformation to data
        raise NotImplementedError
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # reverse the transformation
        raise NotImplementedError
    
class RTMCompose(RTMTransform):
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
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.RTMCompose',
            'transforms': [t.__dict_repr__() for t in self.transforms]
        }
    
    def __repr__(self):
        return f'RTMCompose({self.transforms})'
    
    
class RTMTransformCollection(torch.nn.Module):
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
                    # compose new with existing transform
                    t_new = RTMCompose([t_old, t_new])
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
    
    # def __getitem__(self, name: str):
    #     if name in self.names:
    #         getattr(self, name)
    #     else:
    #         raise ValueError(f'no transform registered under name {name}')
    
    
class NullTransform(RTMTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # do nothing to the data
        return data
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # do nothing to the data
        return data
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.NullTransform'}
    
    def __repr__(self):
        return 'NullTransform()'

class DoubleToFloat(RTMTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # convert to float32 tensor
        return data.to(torch.float)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(torch.double)
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.DoubleToFloat'}
    
    def __repr__(self):
        return 'DoubleToFloat()'
    
class Normalize(RTMTransform):
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
        normalized = (data - self.mean) / self.std
        return normalized
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        # reverse standardization
        # denormalized = data * self.std.to(data.device) + self.mean.to(data.device)
        denormalized = data * self.std + self.mean
        return denormalized
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.Normalize',
            'shape': self.shape
        }
    
    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std})'

class LogTransform(RTMTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return torch.exp(data)
    
    def __dict_repr__(self):
        return {'_target_': 'rtm.rtm_transforms.LogTransform'}
    
    def __repr__(self):
        return 'LogTransform()'
    
class PowerTransform(RTMTransform):
    def __init__(self, exponent: float):
        super().__init__()

        self.register_buffer('exponent', exponent)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.pow(data, self.exponent)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return torch.pow(data, 1/self.exponent)
    
    def __dict_repr__(self):
        return {
            '_target_': 'rtm.rtm_transforms.PowerTransform',
            'exponent': self.exponent
        }
    
    def __repr__(self):
        return f'PowerTransform(exponent={self.exponent})'
    
    
    
def normalization(stats: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Callable:
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

    # if no variation, replace with 1 to avoid division by zero
    std = torch.where(std==0, torch.ones_like(std), std)

    transform = Normalize(mean, std)

    return transform


def load_transforms(dir: str, types=['pixel', 'profile', 'target', 'proxy']):
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
    assert len(transforms) == len(types)
    for i, t in enumerate(types):
        fname = os.path.join(dir, f'{t}_transform.pt')
        torch.save(transforms[i], fname)
    

def numpy_to_tensor(data: np.ndarray, dtype=torch.double) -> torch.Tensor:
    # convert numpy array to Pytorch tensor
    return torch.from_numpy(data).to(dtype)

def tensor_to_numpy(data: torch.Tensor) -> np.ndarray:
    # convert Pytorch tensor to numpy array
    return data.detach().cpu().numpy()
    
