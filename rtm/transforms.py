import numpy as np
import torch
from typing import Union, List, Dict, Callable
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
    
    def __repr__(self):
        return f'RTMCompose({self.transforms})'
    
class DoubleToFloat(RTMTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # convert to float32 tensor
        return data.to(torch.float)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(torch.double)
    
    def __repr__(self):
        return 'DoubleToFloat()'
    
class Normalize(RTMTransform):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
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

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data)
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return torch.exp(data)
    
    def __repr__(self):
        return 'LogTransform()'
    
class FTransform(RTMTransform):
    def __init__(self, f: Callable, f_inv: Callable):
        super().__init__()

        self.f = f
        self.f_inf = f_inv

    def forward(self, data: torch.Tensor) -> torch.Tensor:
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
    
