import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union, List, Dict, Tuple, Optional
import numpy as np

from rtm import utils, layers
from rtm.dimensionality_reduction import DimensionalityReduction, Encoder, Decoder, PCA
from rtm.utils import init_weights


class SurrogateRTM(pl.LightningModule):
    def __init__(self, 
                 target_quantity: str = 'targets',
                 target_transform: utils.RTMTransform = None,
                 eval_scale: torch.Tensor = None, 
                 **kwargs):

        super().__init__()

        self.target_quantity = target_quantity
        self._target_transform = target_transform
        self.eval_scale = eval_scale

        self.save_hyperparameters(ignore=['forward_model', 
                                          'input_dim_reduction', 
                                          'output_dim_reduction'])    

    @property
    def target_transform(self) -> utils.RTMTransform:
        return self._target_transform

    def forward(self, **input_kwargs) -> torch.Tensor:
        """runs RTM prediction"""
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        """training step for neural network models"""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """validation step for neural network models"""
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch[self.target_quantity]
        outputs = self.forward(**batch)

        targets = targets.detach()
        outputs = outputs.detach()

        if self.target_transform:
            # undo normalization and transforms
            targets = self.target_transform.inverse(targets)
            outputs = self.target_transform.inverse(outputs)
        else:
            targets = targets.numpy()
            outputs = outputs.numpy()

        self.outputs.append(outputs)
        self.targets.append(targets)
    
    def on_test_epoch_start(self) -> None:
        """initialize lists to store predictions and targets"""
        self.outputs = []
        self.targets = []
    
    def on_test_epoch_end(self) -> None:

        # process all predictions and targets from the test set
        outputs = np.concatenate(self.outputs, axis=0)
        targets = np.concatenate(self.targets, axis=0)
       
        # compute evaluation metrics, average across input/output dimension
        rmse = utils.rmse(outputs, targets, axis=0).mean()
        nrmse = utils.nrmse(outputs, targets, z=self.eval_scale, axis=0).mean()
        mape = utils.mape(outputs, targets, axis=0).mean()
        mape_continuum = utils.mape_continuum(outputs, targets, axis=0).mean()
        smape = utils.smape(outputs, targets, axis=0).mean()
        rel_bias = utils.relative_bias(outputs, targets, axis=0).mean()
        r2 = utils.r2_score(outputs, targets, axis=0).mean()

        # compute R2 on log-transformed data
        # ensure that outputs are positive
        outputs = np.maximum(outputs, np.finfo(outputs.dtype).eps)
        r2_log = utils.r2_score(np.log(outputs), np.log(targets), axis=0).mean()

        self.log_dict({'test/RMSE': rmse,
                       'test/NRMSE': nrmse,
                       'test/MAPE': mape,
                       'test/MAPE_continuum': mape_continuum,
                       'test/SMAPE': smape,
                       'test/rel_bias': rel_bias,
                       'test/R2': r2,
                       'test/R2_log': r2_log,
                       })

    def predict_step(self, batch, batch_idx):
        """reconstruct input and transform back to physical space"""
        targets = batch[self.target_quantity]
        outputs = self.forward(**batch)

        targets = targets.detach()
        outputs = outputs.detach()

        if self.target_transform:
            # undo normalization and transforms
            targets = self.target_transform.inverse(targets)
            outputs = self.target_transform.inverse(outputs)
        else:
            targets = targets.numpy()
            outputs = outputs.numpy()

        return outputs, targets


    
class NNSurrogateRTM(SurrogateRTM):
    def __init__(self, 
                 forward_model: torch.nn.Module, 
                 lr: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)

        # define forward model
        self.forward_model = forward_model
        
        self.lr = lr
        self.l2_reg = kwargs.get('l2_reg', 0)
        self.loss_scale = kwargs.get('loss_scale', 1.0)

        # list of loss terms making up the overall loss
        self.loss_terms = kwargs.get('loss_terms', [torch.nn.MSELoss()])

    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward_model(inputs)


    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_fct in self.loss_terms:
            loss = loss + loss_fct(outputs, targets)

        # if using SGD without regularization, this is equivalent to scaling the lr
        loss = loss * self.loss_scale

        return loss
    
    def eval_step(self, batch) -> torch.Tensor:
        targets = batch[self.target_quantity]
        outputs = self.forward(**batch)

        loss = self.compute_loss(outputs, targets)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.eval_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True)
        self.log('train/log_loss', torch.log(loss), on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True)

        return loss
            
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step(batch)
        self.log('val/loss', loss, on_epoch=True, logger=True)
        self.log('val/log_loss', torch.log(loss), on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)


class NNSurrogateRTMDiff(NNSurrogateRTM):
    """Surrogate model that predicts the difference between RTM and proxy (non-scattering RTM)"""
    def __init__(self, proxy_quantity: str = 'proxy_targets', **kwargs):
        super().__init__(**kwargs)

        self.proxy_quantity = proxy_quantity

    def prepare_outputs(self, 
                        targets: torch.Tensor, 
                        proxy: torch.Tensor, 
                        outputs: torch.Tensor
                        ) -> Tuple[np.ndarray, np.ndarray]:

        targets = targets.detach()
        proxy = proxy.detach()
        outputs = outputs.detach()

        if self.target_transform:
            # undo normalization and transforms
            targets = self.target_transform.inverse(targets)
            proxy = self.target_transform.inverse(proxy)
            outputs = self.target_transform.inverse(outputs)
        else:
            targets = targets.numpy()
            proxy = proxy.numpy()
            outputs = outputs.numpy()

        # compute actual spectra
        outputs = proxy + outputs
        targets = proxy + targets

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch[self.target_quantity]
        proxy = batch[self.proxy_quantity]
        outputs = self.forward(**batch)

        outputs, targets = self.prepare_outputs(targets, proxy, outputs)

        self.outputs.append(outputs)
        self.targets.append(targets)

    def predict_step(self, batch, batch_idx):
        """make prediction"""
        targets = batch[self.target_quantity]
        proxy = batch[self.proxy_quantity]
        outputs = self.forward(**batch)

        outputs, targets = self.prepare_outputs(targets, proxy, outputs)

        return outputs, targets


  

class LatentNNSurrogateRTM(NNSurrogateRTM):
    def __init__(self, 
                 input_dim_reduction: Union[DimensionalityReduction, Encoder] = None, 
                 output_dim_reduction: Union[DimensionalityReduction, Decoder] = None,
                 proxy_module: Optional[torch.nn.Module] = None,
                 nonlinearity: Optional[str] = None, 
                 **kwargs):
        super().__init__(**kwargs)

        # model components
        self.input_dim_reduction = input_dim_reduction
        self.output_dim_reduction = output_dim_reduction
        self.proxy_module = proxy_module

        print(proxy_module)

        self.nonlinearity = getattr(F, nonlinearity) if nonlinearity else None

        # # TODO: add option to freeze all but last layer? (for end-to-end fine-tuning)
        # if kwargs.get('pretrained_dim_reduction', False):
        #     # only learn surrogate model parameters
        #     utils.freeze_weights(self.input_dim_reduction)
        #     utils.freeze_weights(self.output_dim_reduction)

        self.latent_loss = kwargs.get('latent_loss', False)

    @property
    def target_transform(self) -> utils.RTMTransform:
        return self._target_transform
    
    @target_transform.setter
    def target_transform(self, target_transform: utils.RTMTransform):
        self._target_transform = target_transform
        if hasattr(self.input_dim_reduction, 'target_transform'):
            self.input_dim_reduction.target_transform = target_transform
        if hasattr(self.output_dim_reduction, 'target_transform'):
            self.output_dim_reduction.target_transform = target_transform
    
    def forward(self, inputs, **kwargs) -> torch.Tensor:

        if isinstance(self.input_dim_reduction, (DimensionalityReduction, Encoder)):
            inputs = self.input_dim_reduction.encode(inputs)

            # if neural network encoder, apply non-linearity before passing on to forward model
            if isinstance(self.input_dim_reduction, Encoder) and self.nonlinearity:
                inputs = self.nonlinearity(inputs)

        if self.proxy_module:
            # extract information from proxy spectra and append to other inputs
            proxy = self.proxy_module(kwargs['proxy_targets'])
            inputs = torch.cat([inputs, proxy], dim=-1)

        outputs = self.forward_model(inputs)

        # if neural network decoder, apply non-linearity before passing on to forward model
        if isinstance(self.output_dim_reduction, Decoder) and self.nonlinearity:
            outputs = self.nonlinearity(outputs)

        if isinstance(self.output_dim_reduction, (DimensionalityReduction, Decoder)):
            outputs = self.output_dim_reduction.decode(outputs)

        return outputs
    
    def eval_step(self, batch) -> torch.Tensor:
        targets = batch[self.target_quantity]

        outputs = self.forward(**batch)

        # inputs = batch['inputs']
        
        # if isinstance(self.input_dim_reduction, (DimensionalityReduction, Encoder)):
        #     inputs = self.input_dim_reduction.encode(inputs)

        # outputs = self.forward_model(inputs)

        if isinstance(self.output_dim_reduction, (DimensionalityReduction, Decoder)):
            if self.latent_loss:
                targets = self.output_dim_reduction.encode(targets)
                outputs = self.output_dim_reduction.encode(outputs)
            # else:
            #     outputs = self.output_dim_reduction.decode(outputs)
            
        loss = self.compute_loss(outputs, targets)

        return loss
    
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)
    
    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if isinstance(self.input_dim_reduction, PCA):
            checkpoint['input_dim_reduction'] = self.input_dim_reduction.pca_dict()
        if isinstance(self.output_dim_reduction, PCA):
            checkpoint['output_dim_reduction'] = self.output_dim_reduction.pca_dict()
        checkpoint['target_transform'] = self.target_transform

    def on_load_checkpoint(self, checkpoint):
        print(f'Loading model state on device {self.device}...')
        self.load_model_state(checkpoint, map_location=self.device)
        
    def load_model_state(self, checkpoint, map_location=None):
        self.load_state_dict(checkpoint['state_dict'])

        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction.load_pca_dict(checkpoint['input_dim_reduction'], map_location)
            
        if isinstance(self.output_dim_reduction, PCA):
            self.output_dim_reduction.load_pca_dict(checkpoint['output_dim_reduction'], map_location)

        self.target_transform = checkpoint['target_transform']

        # make sure everything is on the same device
        self.to(map_location)

    def _apply(self, convert):
        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction._apply(convert)
        if isinstance(self.output_dim_reduction, PCA):
            self.output_dim_reduction._apply(convert)
        
        return super()._apply(convert)
        
    
class LatentNNSurrogateRTMDiff(LatentNNSurrogateRTM):
    """Surrogate model that predicts the difference between RTM and proxy (non-scattering RTM)"""
    def __init__(self, proxy_quantity: str = 'proxy_targets', **kwargs):
        super().__init__(**kwargs)

        self.proxy_quantity = proxy_quantity

    def prepare_outputs(self, 
                        targets: torch.Tensor, 
                        proxy: torch.Tensor, 
                        outputs: torch.Tensor
                        ) -> Tuple[np.ndarray, np.ndarray]:

        targets = targets.detach()
        proxy = proxy.detach()
        outputs = outputs.detach()

        if self.target_transform:
            # undo normalization and transforms
            targets = self.target_transform.inverse(targets)
            proxy = self.target_transform.inverse(proxy)
            outputs = self.target_transform.inverse(outputs)
        else:
            targets = targets.numpy()
            proxy = proxy.numpy()
            outputs = outputs.numpy()

        # compute actual spectra
        outputs = proxy + outputs
        targets = proxy + targets

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch[self.target_quantity]
        proxy = batch[self.proxy_quantity]
        outputs = self.forward(**batch)

        outputs, targets = self.prepare_outputs(targets, proxy, outputs)

        self.outputs.append(outputs)
        self.targets.append(targets)

    def predict_step(self, batch, batch_idx):
        """make prediction"""
        targets = batch[self.target_quantity]
        proxy = batch[self.proxy_quantity]
        outputs = self.forward(**batch)

        outputs, targets = self.prepare_outputs(targets, proxy, outputs)

        return outputs, targets



class NonScatteringRTM(SurrogateRTM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, **input_kwargs) -> torch.Tensor:
        return input_kwargs['proxy_targets']


        
class HybridRTM(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, **kwargs):
        super().__init__()

        self.mlp = MLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out)

    def forward(self, inputs, transmission, **kwargs) -> torch.Tensor:
        # use NN to predict difference between RTM and proxy
        offset = self.mlp(inputs)
        output = transmission + offset

        return output
    


class MLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, **kwargs):
        super(MLP, self).__init__()

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_layers = kwargs.get('n_fc_layers', 1)

        self.fc_in = torch.nn.Linear(n_in, n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden)
                                        for _ in range(self.n_layers - 1)])
        self.fc_out = torch.nn.Linear(n_hidden, n_out)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)

    def forward(self, features):

        x = F.relu(self.fc_in(features))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = F.relu(l(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)

        return x
    

class MLP_v2(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, **kwargs):
        super().__init__()

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_layers = kwargs.get('n_fc_layers', 1)

        nonlinearity = kwargs.get('nonlinearity', 'relu')
        self.nonlinearity = getattr(F, nonlinearity)

        self.fc_in = torch.nn.Linear(n_in, n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden)
                                        for _ in range(self.n_layers - 1)])
        self.fc_out = torch.nn.Linear(n_hidden, n_out)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)

    def forward(self, features):

        x = self.nonlinearity(self.fc_in(features))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = self.nonlinearity(l(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)

        return x
    

