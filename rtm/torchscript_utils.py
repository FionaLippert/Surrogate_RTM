import torch
import os
from typing import Union, Tuple, Optional, Dict
from torch.fx.experimental.proxy_tensor import make_fx

"""
This is a collection of functions to create, save, and load TorchScripts 
from trained PyTorch models
"""

def script_model(
        model: torch.nn.Module, 
        device_type='cpu') -> torch.jit.ScriptModule:
    """
    Script a PyTorch model using torch.jit.script and return a TorchScript model.

    Note that this does not work for certain models that use dynamic control flow 
    or other features that are not compatible with scripting.
    In such cases, use tracing instead.

    Args:
        model: The PyTorch model to be scripted.
        device_type: The device type to use for scripting ('cpu' or 'cuda').
    Returns:
        A TorchScript model obtained by scripting the original model.
    """
    
    # move model to device
    device = torch.device(device_type)
    model = model.to(device)

    # make sure model is in eval mode
    model.eval()

    # convert model to TorchScript using scripting
    scripted_model = torch.jit.script(model)

    return scripted_model

def trace_model(
        model: torch.nn.Module,
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]],
        check_tolerance: Optional[float] = 1e-5,
        device_type='cpu') -> torch.jit.ScriptModule:
    """
    Trace a PyTorch model using torch.jit.trace and return a TorchScript model.

    Note that this only includes operations that are executed during the forward 
    pass with the provided dummy input. 

    Args:
        model: The PyTorch model to be traced.
        dummy_input: A sample input tensor (or tuple/dict of tensors) to the model.
        check_tolerance: Tolerance for checking the traced model against the original model.
        device_type: The device type to use for tracing ('cpu' or 'cuda').
    Returns:
        A TorchScript model obtained by tracing the original model.
    """
    
    # move model to device
    device = torch.device(device_type)
    model = model.to(device)

    # make sure model is in eval mode
    model.eval()

    if isinstance(dummy_input, torch.Tensor):
        # move inputs to device
        dummy_input = dummy_input.to(device)
        # convert model to TorchScript using tracing
        traced_model = torch.jit.trace(model, 
                                       example_inputs=dummy_input, 
                                       check_tolerance=check_tolerance)
    elif isinstance(dummy_input, tuple):
        # input args
        dummy_input_new = []
        for v in dummy_input:
            # move inputs to device
            dummy_input_new.append(v.to(device))
        # convert model to TorchScript using tracing
        traced_model = torch.jit.trace(model, 
                                       example_inputs=tuple(dummy_input_new),
                                       check_tolerance=check_tolerance)
    elif isinstance(dummy_input, dict):
        # input kwargs
        dummy_input_new = {}
        for k, v in dummy_input.items():
            # move inputs to device
            dummy_input_new[k] = v.to(device)
        # convert model to TorchScript using tracing
        traced_model = torch.jit.trace(model, 
                                       example_kwarg_inputs=dummy_input_new,
                                       check_tolerance=check_tolerance)
    else:
        raise ValueError('dummy_input must be a torch.Tensor, tuple, or dict')

    # freeze model (inline submodules, etc to optimize performance)
    traced_model = torch.jit.freeze(traced_model)

    return traced_model


def script_model_and_save(
        model: torch.nn.Module, 
        filepath: str, 
        device_type='cpu'):
    """
    Script a PyTorch model and save it to a file.
    """

    # convert model to TorchScript using scripting
    scripted_model = script_model(model, device_type)

    # save scripted model
    scripted_model.save(filepath)


def trace_model_and_save(
        model: torch.nn.Module, 
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]],
        filepath: str,
        check_tolerance: Optional[float] = 1e-5,
        device_type='cpu'):
    """
    Trace a PyTorch model and save it to a file.
    """
    
    traced_model = trace_model(model, dummy_input, check_tolerance, device_type)

    # save scripted model
    traced_model.save(filepath)

def load_torchscript(filepath: str) -> torch.nn.Module:
    # load TorchScript from file
    model = torch.jit.load(filepath)

    return model


def trace_jacobian(traced_model: torch.jit.ScriptModule, 
                   dummy_input: Tuple[torch.Tensor], 
                   check_tolerance: Optional[float] = 1e-5):
    """
    Create a TorchScript-compatible Jacobian function for a PyTorch model
    using torch.func.jacrev and make_fx.
    
    Args:
        model: TorchScript model created using torch.jit.trace
        example_inputs: Example input tensor for tracing
        args: Additional arguments for the model forward function (e.g. proxy_targest or profile_inputs)
    
    Returns:
        Traced Jacobian function that can be used in TorchScript
    """
    # Create the Jacobian function using jacrev
    jac_func = torch.func.jacrev(traced_model.forward, argnums=0)
    
    # The key part: apply make_fx to jac_func and then immediately call it with inputs
    # This creates an FX graph that captures the computation for these specific inputs
    fx_jac = make_fx(jac_func)(*dummy_input)
    
    # Now we can trace this FX graph
    traced_jac = torch.jit.trace(fx_jac, 
                                 example_inputs=dummy_input, 
                                 check_tolerance=check_tolerance)
    
    return traced_jac