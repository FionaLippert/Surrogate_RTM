import torch

##################################################
# Custom loss functions
##################################################

class WeightedMAE(torch.nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        # weight applied to scale loss
        self.weight = weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.abs((outputs - targets)))

        return self.weight * loss

class WeightedMSE(torch.nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        # weight applied to scale loss
        self.weight = weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.square((outputs - targets)))

        return self.weight * loss
    
class RelativeMSE(torch.nn.Module):
    def __init__(self, eps: float = 0.0) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.square((outputs - targets) / (targets + self.eps)))

        return loss

class ContinuumWeightedMAE(torch.nn.Module):
    def __init__(self, weight: float = 1.0, log_transformed: bool = False) -> None:
        super().__init__()

        # weight applied to scale loss
        self.weight = weight
        self.log_transformed = log_transformed

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MAE scaled by continuum of each individual spectrum (i.e. max across wavelengths)
        continuum = targets.max(1, keepdims=True).values

        residuals = outputs - targets
        if self.log_transformed:
            # for log-transformed radiances, division by the continuum becomes a difference
            residuals = residuals - continuum
        else:
            residuals = residuals / continuum
        
        loss = torch.mean(torch.abs(residuals))

        return self.weight * loss

class ContinuumWeightedMSE(torch.nn.Module):
    def __init__(self, weight: float = 1.0, log_transformed: bool = False) -> None:
        super().__init__()

        # weight applied to scale loss
        self.weight = weight
        self.log_transformed = log_transformed

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE scaled by continuum of each individual spectrum (i.e. max across wavelengths)
        continuum = targets.max(1, keepdims=True).values

        residuals = outputs - targets
        if self.log_transformed:
            # for log-transformed radiances, division by the continuum becomes a difference
            residuals = residuals - continuum
        else:
            residuals = residuals / continuum

        loss = torch.mean(torch.square(residuals))

        return self.weight * loss
        

class TotalRadianceMAE(torch.nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        # weight applied to scale loss
        self.weight = weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # encourage that total predicted radiance matches total target radiance
        continuum = targets.max(1, keepdims=False).values
        output_total_radiance = torch.trapz(outputs, dim=1) / continuum
        target_total_radiance = torch.trapz(targets, dim=1) / continuum
        loss = torch.mean(torch.abs(output_total_radiance -  target_total_radiance))

        return self.weight * loss

