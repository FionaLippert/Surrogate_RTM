import torch
import os
from typing import Union, Tuple, Optional, Dict

"""
This is a collection of functions to create, save, and load TorchScripts from trained PyTorch models
"""

def script_model_and_save(
        model: torch.nn.Module, 
        filepath: str, 
        device_type='cpu'):
    
    # move model to device
    device = torch.device(device_type)
    model = model.to(device)

    # make sure model is in eval mode
    model.eval()

    # convert model to TorchScript using scripting
    scripted_model = torch.jit.script(model)

    # save scripted model
    scripted_model.save(filepath)


def trace_model_and_save(
        model: torch.nn.Module, 
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        filepath: str,
        device_type='cpu'):
    
    # move model to device
    device = torch.device(device_type)
    model = model.to(device)

    # make sure model is in eval mode
    model.eval()

    if isinstance(dummy_input, torch.Tensor):
        # move inputs to device
        dummy_input = dummy_input.to(device)
        # convert model to TorchScript using tracing
        traced_model = torch.jit.trace(model, example_inputs=dummy_input)
    else:
        # input kwargs
        for k, v in dummy_input.items():
            # move inputs to device
            dummy_input[k] = v.to(device)
        # convert model to TorchScript using tracing
        traced_model = torch.jit.trace(model, example_kwarg_inputs=dummy_input)

    # freeze model (inline submodules, etc to optimize performance)
    traced_model = torch.jit.freeze(traced_model)

    # save scripted model
    traced_model.save(filepath)

def load_torchscript(filepath: str) -> torch.nn.Module:
    # load TorchScript from file
    model = torch.jit.load(filepath)

    return model