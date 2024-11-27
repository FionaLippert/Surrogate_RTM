from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch

class GradNormCallback(Callback):

    def on_after_backward(self, trainer, pl_module):
        total_norm, max_norm = gradient_norm(pl_module)
        pl_module.log("model/total_grad_norm", total_norm)
        pl_module.log("model/max_grad_norm", max_norm)

def gradient_norm(model):
    total_norm = 0.0
    max_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            max_norm = max(max_norm, param_norm)
    total_norm = total_norm ** 0.5

    return total_norm, max_norm