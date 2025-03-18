import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, ConstantLR
from typing import Callable, Union, List, Dict, Tuple, Optional
import numpy as np

from rtm import rtm_transforms, utils, layers
from rtm.dimensionality_reduction import DimensionalityReduction, Encoder, Decoder, PCA
from rtm.utils import init_weights


class SurrogateRTM(pl.LightningModule):
    def __init__(self, 
                 input_quantity: str = 'inputs',
                 target_quantity: str = 'targets',
                 proxy_quantity: str = 'proxy_targets',
                 transforms: rtm_transforms.RTMTransformCollection = None,
                #  input_transform: rtm_transforms.RTMTransform = None,
                #  target_transform: rtm_transforms.RTMTransform = None,
                #  proxy_transform: rtm_transforms.RTMTransform = None,
                 eval_scale: torch.Tensor = None, 
                 **kwargs):

        super().__init__()

        self.input_quantity = input_quantity
        self.target_quantity = target_quantity
        self.proxy_quantity = proxy_quantity

        self.transforms = transforms

        # self.input_transform = input_transform
        # self.target_transform = target_transform
        # self.proxy_transform = proxy_transform
        
        self.eval_scale = eval_scale

        self.save_hyperparameters(ignore=['forward_model', 
                                          'input_dim_reduction', 
                                          'output_dim_reduction',
                                          'transforms'])    

    @property
    def input_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'pixel_transform', None)
    
    @input_transform.setter
    def input_transform(self, input_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'pixel_transform': input_transform})
        else:
            raise Warning('Trying to set input_transform for a model without transforms')

    @property
    def target_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'target_transform', None)
    
    @target_transform.setter
    def target_transform(self, target_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'target_transform': target_transform})
        else:
            raise Warning('Trying to set target_transform for a model without transforms')

    @property
    def proxy_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'proxy_transform', None)
    
    @proxy_transform.setter
    def proxy_transform(self, proxy_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'proxy_transform': proxy_transform})
        else:
            raise Warning('Trying to set proxy_transform for a model without transforms')

    def forward(self, **input_kwargs) -> torch.Tensor:
        """runs RTM prediction"""
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        """training step for neural network models"""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """validation step for neural network models"""
        raise NotImplementedError
    
    def prepare_outputs(self, 
                        outputs: torch.Tensor, 
                        targets: torch.Tensor
                        ) -> Tuple[np.ndarray, np.ndarray]:
        
        outputs = outputs.detach()
        targets = targets.detach()

        # if self.target_transform:
        #     # undo normalization and transforms
        #     outputs = self.target_transform.inverse(outputs)
        #     targets = self.target_transform.inverse(targets)

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        outputs = self.forward(**batch)
        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
        outputs, targets = self.prepare_outputs(outputs, targets)

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
        rmse_continuum = utils.rmse_continuum(outputs, targets, axis=0).mean()
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
                       'test/RMSE_continuum': rmse_continuum,
                       'test/SMAPE': smape,
                       'test/rel_bias': rel_bias,
                       'test/R2': r2,
                       'test/R2_log': r2_log,
                       })

    def predict_step(self, batch, batch_idx):
        """reconstruct input and transform back to physical space"""
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        outputs = self.forward(**batch)
        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
        outputs, targets = self.prepare_outputs(outputs, targets)

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

        # learning rate scheduling
        self.lr_schedulers = kwargs.get('lr_schedulers', None)
        self.lr_milestones = kwargs.get('lr_milestones', None)
        self.lr_scheduler_interval = kwargs.get('lr_scheduler_interval', 'epoch')

    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:

        # prepare inputs
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        # prepare proxy
        if self.proxy_quantity in kwargs and self.proxy_transform is not None:
            kwargs[self.proxy_quantity] = self.proxy_transform(kwargs[self.proxy_quantity])

        return self.forward_model(inputs, **kwargs)


    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_fct in self.loss_terms:
            loss = loss + loss_fct(outputs, targets)

        # if using SGD without regularization, this is equivalent to scaling the lr
        loss = loss * self.loss_scale

        return loss
    
    def eval_step(self, batch) -> torch.Tensor:
        # prepare targets
        targets = batch[self.target_quantity]
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        # make prediction
        outputs = self.forward(**batch)

        # compute loss function
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        if self.lr_schedulers is not None and self.lr_milestones is not None:
            scheduler = SequentialLR(optimizer, 
                                     [s_partial(optimizer) for s_partial in self.lr_schedulers], 
                                     milestones=self.lr_milestones)
        else:
            # constant learning rate
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1e8)

        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': self.lr_scheduler_interval
        }

        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config}
    
    # transforms are torch.nn.Module objects now and dont' need to be saved separately
    # def on_save_checkpoint(self, checkpoint):
    #     # save additional objects to checkpoint 
    #     checkpoint['target_transform'] = self.target_transform
    #     checkpoint['proxy_transform'] = self.proxy_transform
    
    def on_load_checkpoint(self, checkpoint):
        print(f'Loading model state on device {self.device}...')
        self.load_model_state(checkpoint, map_location=self.device)
        
    def load_model_state(self, checkpoint, map_location=None):
        # load standard pytorch state_dict
        self.load_state_dict(checkpoint['state_dict'])

        # # load additional objects from checkpoint
        # self.target_transform = checkpoint['target_transform']
        # self.proxy_transform = checkpoint['proxy_transform']

        # make sure everything is on the same device
        self.to(map_location)


class NNSurrogateRTMDiff(NNSurrogateRTM):
    """Surrogate model that predicts the difference between RTM and proxy (non-scattering RTM)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_outputs(self, 
                        outputs: torch.Tensor,
                        targets: torch.Tensor, 
                        proxy: torch.Tensor, 
                        ) -> Tuple[np.ndarray, np.ndarray]:

        outputs = outputs.detach()
        targets = targets.detach()
        proxy = proxy.detach()

        # if self.target_transform:
        #     # undo normalization and transforms
        #     outputs = self.target_transform.inverse(outputs)
        #     targets = self.target_transform.inverse(targets)

        # if self.proxy_transform:
        #     proxy = self.proxy_transform.inverse(proxy)

        # compute actual spectra
        outputs = proxy + outputs
        targets = proxy + targets

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        # prepare targets
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        # prepare proxy
        proxy = batch[self.proxy_quantity]
        # if self.proxy_transform is not None:
        #     proxy = self.proxy_transform(proxy)

        # make prediction
        outputs = self.forward(**batch)
        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
        outputs, targets = self.prepare_outputs(outputs, targets, proxy)

        self.outputs.append(outputs)
        self.targets.append(targets)

    def predict_step(self, batch, batch_idx):
        """make prediction"""
        # prepare targets
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        # prepare proxy
        proxy = batch[self.proxy_quantity]
        # if self.proxy_transform is not None:
        #     proxy = self.proxy_transform(proxy)
        
        # make prediction
        outputs = self.forward(**batch)
        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
        outputs, targets = self.prepare_outputs(outputs, targets, proxy)

        return outputs, targets
    
class NNSurrogateRTMRatio(NNSurrogateRTM):
    """Surrogate model that predicts the ratio between RTM and proxy (non-scattering RTM)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_outputs(self, 
                        outputs: torch.Tensor,
                        targets: torch.Tensor, 
                        proxy: torch.Tensor, 
                        ) -> Tuple[np.ndarray, np.ndarray]:

        outputs = outputs.detach()
        targets = targets.detach()
        proxy = proxy.detach()

        # compute actual spectra
        outputs = proxy * outputs
        targets = proxy * targets

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        # prepare targets
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        # prepare proxy
        proxy = batch[self.proxy_quantity]
        # if self.proxy_transform is not None:
        #     proxy = self.proxy_transform(proxy)

        # make prediction
        outputs = self.forward(**batch)
        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
        outputs, targets = self.prepare_outputs(outputs, targets, proxy)

        self.outputs.append(outputs)
        self.targets.append(targets)

    def predict_step(self, batch, batch_idx):
        """make prediction"""
        # prepare targets
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        # prepare proxy
        proxy = batch[self.proxy_quantity]
        # if self.proxy_transform is not None:
        #     proxy = self.proxy_transform(proxy)
        
        # make prediction
        outputs = self.forward(**batch)
        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
        outputs, targets = self.prepare_outputs(outputs, targets, proxy)

        return outputs, targets


class LatentNNForwardModel(torch.nn.Module):
    """
    Simple forward model that first encodes to the input features to a 
    lower-dimensional latent space (e.g. using PCA or autoencoder),
    then passes them through an MLP, and finally applies a decoder 
    to obtain the final output.
    """
    def __init__(self,  
                 core_nn: torch.nn.Module, 
                 proxy_nn: Optional[torch.nn.Module] = None,
                 proxy_quantity: Optional[str] = 'proxy_targets',
                 encoder: Optional[Encoder] = None, 
                 decoder: Optional[Decoder] = None,
                 **kwargs):
        super().__init__()

        self.core_nn = core_nn

        # optional proxy processing (e.g. non-scattering radiances)
        self.proxy_nn = proxy_nn
        self.proxy_quantity = proxy_quantity

        # optional encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, 
                inputs: torch.Tensor, 
                **kwargs
                ) -> torch.Tensor:
        # predicts radiance spectrum from input features, and optionally proxy spectra
        
        if self.encoder is not None:
            inputs = self.encoder.encode(inputs)

        if self.proxy_nn is not None:
            proxy = self.proxy_nn(kwargs[self.proxy_quantity])
            inputs = torch.cat([inputs, proxy], dim=-1)

        outputs = self.core_nn(inputs)

        if self.decoder is not None:
            outputs = self.decoder.decode(outputs)

        return outputs



class SequenceNNForwardModel(torch.nn.Module):
    def __init__(self, 
                 sequence_length: int,
                 encoder: Optional[Encoder] = None,
                 pos_encoding: Optional[torch.nn.Module] = None,
                 proxy_module: Optional[torch.nn.Module] = None,
                 proxy_quantity: Optional[str] = 'proxy_targets',
                 profile_module: Optional[torch.nn.Module] = None,
                 profile_quantity: Optional[str] = 'profile_inputs',
                 input_processing: Optional[torch.nn.Module] = None,
                 use_proxy_input: bool = False,
                 use_profile_input: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length

        self.encoder = encoder
        self.pos_encoding = pos_encoding
        self.input_processing = input_processing

        self.proxy_module = proxy_module
        self.use_proxy_input = use_proxy_input
        self.proxy_quantity = proxy_quantity

        self.profile_module = profile_module
        self.use_profile_input = use_profile_input
        self.profile_quantity = profile_quantity


    def forward(self, inputs, **kwargs) -> torch.Tensor:

        if self.encoder is not None:
            inputs = self.encoder.encode(inputs)

        if self.use_profile_input:
            # profile inputs have shape [batch_size, n_layers, n_vars]
            profile_input = kwargs[self.profile_quantity]

            if self.profile_module is not None:
                profile_input = self.profile_module(profile_input)

            # make sure profile features are flattened
            profile_input = profile_input.reshape(inputs.size(0), -1)
            
            # add profile information to inputs
            inputs = torch.cat([inputs, profile_input], dim=1)

        # inputs have shape [batch_size, n_features] --> expand across wavelengths
        inputs = inputs.unsqueeze(1).expand(-1, self.sequence_length, -1)

        if self.use_proxy_input:
            # proxy targets have shape [batch_size, n_wavelengths]
            # expand inputs and add them as additional channels to proxy spectra
            proxy = kwargs[self.proxy_quantity].unsqueeze(-1)

            if self.proxy_module:
                proxy = self.proxy_module(proxy)

            # add proxy information to inputs
            inputs = torch.cat([inputs, proxy], dim=-1)

        if self.input_processing:
            inputs = self.input_processing(inputs)

        if self.pos_encoding:
            # add positional information to inputs
            inputs = self.pos_encoding(inputs)

        # make prediction
        outputs = self.forward_model(inputs)
        outputs = outputs.reshape(-1, self.sequence_length)

        return outputs

        

  

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

        self.nonlinearity = getattr(F, nonlinearity) if nonlinearity else None

        # # TODO: add option to freeze all but last layer? (for end-to-end fine-tuning)
        # if kwargs.get('pretrained_dim_reduction', False):
        #     # only learn surrogate model parameters
        #     utils.freeze_weights(self.input_dim_reduction)
        #     utils.freeze_weights(self.output_dim_reduction)

        # self.latent_loss = kwargs.get('latent_loss', False)

    @property
    def target_transform(self) -> rtm_transforms.RTMTransform:
        return self._target_transform
    
    @target_transform.setter
    def target_transform(self, target_transform: rtm_transforms.RTMTransform):
        self._target_transform = target_transform
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

    
    # def validation_step(self, batch, batch_idx):
    #     return super().validation_step(batch, batch_idx)
    
    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if isinstance(self.input_dim_reduction, PCA):
            checkpoint['input_dim_reduction'] = self.input_dim_reduction.pca_dict()
        if isinstance(self.output_dim_reduction, PCA):
            checkpoint['output_dim_reduction'] = self.output_dim_reduction.pca_dict()

        super().on_save_checkpoint(checkpoint)
        
        
    def load_model_state(self, checkpoint, map_location=None):
        # load PCA parameters
        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction.load_pca_dict(checkpoint['input_dim_reduction'], map_location)
        if isinstance(self.output_dim_reduction, PCA):
            self.output_dim_reduction.load_pca_dict(checkpoint['output_dim_reduction'], map_location)

        super().load_model_state(checkpoint, map_location=map_location)

    def _apply(self, convert):
        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction._apply(convert)
        if isinstance(self.output_dim_reduction, PCA):
            self.output_dim_reduction._apply(convert)
        
        return super()._apply(convert)
        
    
class LatentNNSurrogateRTMDiff(LatentNNSurrogateRTM):
    """Surrogate model that predicts the difference between RTM and proxy (non-scattering RTM)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @property
    # def proxy_transform(self) -> utils.RTMTransform:
    #     return self._proxy_transform
    
    # @proxy_transform.setter
    # def proxy_transform(self, proxy_transform: utils.RTMTransform):
    #     self._proxy_transform = proxy_transform

    def prepare_outputs(self, 
                        outputs: torch.Tensor,
                        targets: torch.Tensor, 
                        proxy: torch.Tensor, 
                        ) -> Tuple[np.ndarray, np.ndarray]:

        outputs = outputs.detach()
        targets = targets.detach()
        proxy = proxy.detach()

        if self.target_transform:
            # undo normalization and transforms
            outputs = self.target_transform.inverse(outputs)
            targets = self.target_transform.inverse(targets)

        if self.proxy_transform:
            proxy = self.proxy_transform.inverse(proxy)

        # compute actual spectra
        outputs = proxy + outputs
        targets = proxy + targets

        outputs = outputs.numpy()
        targets = targets.numpy()

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch[self.target_quantity]
        proxy = batch[self.proxy_quantity]
        outputs = self.forward(**batch)

        outputs, targets = self.prepare_outputs(outputs, targets, proxy)

        self.outputs.append(outputs)
        self.targets.append(targets)

    def predict_step(self, batch, batch_idx):
        """make prediction"""
        targets = batch[self.target_quantity]
        proxy = batch[self.proxy_quantity]
        outputs = self.forward(**batch)

        outputs, targets = self.prepare_outputs(outputs, targets, proxy)

        return outputs, targets
    

class SequenceNNSurrogateRTMDiff(NNSurrogateRTMDiff):
    def __init__(self, 
                 output_dim: int,
                 input_dim_reduction: Optional[Union[DimensionalityReduction, Encoder]] = None,
                 proxy_module: Optional[torch.nn.Module] = None, 
                 pos_encoding: Optional[torch.nn.Module] = None,
                 input_processing: Optional[torch.nn.Module] = None,
                 use_proxy_input: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_dim_reduction = input_dim_reduction

        # note that the output dim of proxy_module and pos_encoding need to match
        self.proxy_module = proxy_module
        self.pos_encoding = pos_encoding
        self.input_processing = input_processing

        self.use_proxy_input = use_proxy_input
        self.n_wavelengths = output_dim


    # def forward(self, inputs, **kwargs) -> torch.Tensor:

    #     if self.use_proxy_input:
    #         # proxy targets have shape [batch_size, n_wavelengths]
    #         # expand inputs and add them as additional channels to proxy spectra
    #         proxy = kwargs[self.proxy_quantity].unsqueeze(-1)

    #         if self.proxy_module:
    #             proxy = self.proxy_module(proxy)

    #         # if self.pos_encoding:
    #             # pos encoding requires shape [batch_size, n_wavelengths, n_features]
    #             # proxy = self.pos_encoding(proxy)

    #     if isinstance(self.input_dim_reduction, (DimensionalityReduction, Encoder)):
    #         inputs = self.input_dim_reduction.encode(inputs)

    #     # inputs have shape [batch_size, n_features]
    #     # expand across wavelengths
    #     inputs = inputs.unsqueeze(1).expand(-1, self.n_wavelengths, -1)

    #     if self.pos_encoding:
    #         # add positional information to inputs
    #         inputs = self.pos_encoding(inputs)

    #     if self.use_proxy_input:
    #         inputs = torch.cat([inputs, proxy], dim=-1)

    #     # forward model works on a sequence of shape (batch_size, n_wavelengths, n_channels)
    #     outputs = self.forward_model(inputs).reshape(inputs.size(0), self.n_wavelengths)

    #     return outputs

    def forward(self, inputs, **kwargs) -> torch.Tensor:

        if self.use_proxy_input:
            # proxy targets have shape [batch_size, n_wavelengths]
            # expand inputs and add them as additional channels to proxy spectra
            proxy = kwargs[self.proxy_quantity].unsqueeze(-1)

            if self.proxy_module:
                proxy = self.proxy_module(proxy)

        if isinstance(self.input_dim_reduction, (DimensionalityReduction, Encoder)):
            inputs = self.input_dim_reduction.encode(inputs)

        # inputs have shape [batch_size, n_features]
        # expand across wavelengths
        inputs = inputs.unsqueeze(1).expand(-1, self.n_wavelengths, -1)

        if self.use_proxy_input:
            inputs = torch.cat([inputs, proxy], dim=-1)

        if self.input_processing:
            inputs = self.input_processing(inputs)

        if self.pos_encoding:
            # add positional information to inputs
            inputs = self.pos_encoding(inputs)

        outputs = self.forward_model(inputs).reshape(-1, self.n_wavelengths)

        return outputs
    
    
    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if isinstance(self.input_dim_reduction, PCA):
            checkpoint['input_dim_reduction'] = self.input_dim_reduction.pca_dict()
        super().on_save_checkpoint(checkpoint)
        
    def load_model_state(self, checkpoint, map_location=None):

        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction.load_pca_dict(checkpoint['input_dim_reduction'], map_location)
            
        super().load_model_state(checkpoint, map_location=map_location)

    def _apply(self, convert):
        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction._apply(convert)
        
        return super()._apply(convert)
    

class SequenceNNSurrogateRTMDiff_v2(NNSurrogateRTM):
    def __init__(self, 
                 output_dim: int,
                 input_dim_reduction: Optional[Union[DimensionalityReduction, Encoder]] = None,
                 proxy_module: Optional[torch.nn.Module] = None, 
                 pos_encoding: Optional[torch.nn.Module] = None,
                 input_processing: Optional[torch.nn.Module] = None,
                 use_proxy_input: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_dim_reduction = input_dim_reduction

        # note that the output dim of proxy_module and pos_encoding need to match
        self.proxy_module = proxy_module
        self.pos_encoding = pos_encoding
        self.input_processing = input_processing

        self.use_proxy_input = use_proxy_input
        self.n_wavelengths = output_dim



    def forward(self, inputs, **kwargs) -> torch.Tensor:

        proxy = kwargs[self.proxy_quantity]

        if self.use_proxy_input:
            # proxy targets have shape [batch_size, n_wavelengths]
            # expand inputs and add them as additional channels to proxy spectra
            proxy_input = proxy.unsqueeze(-1)

            if self.proxy_module:
                proxy_input = self.proxy_module(proxy_input)

        if isinstance(self.input_dim_reduction, (DimensionalityReduction, Encoder)):
            inputs = self.input_dim_reduction.encode(inputs)

        # inputs have shape [batch_size, n_features]
        # expand across wavelengths
        inputs = inputs.unsqueeze(1).expand(-1, self.n_wavelengths, -1)

        if self.use_proxy_input:
            inputs = torch.cat([inputs, proxy_input], dim=-1)

        if self.input_processing:
            inputs = self.input_processing(inputs)

        if self.pos_encoding:
            # add positional information to inputs
            inputs = self.pos_encoding(inputs)

        outputs = self.forward_model(inputs).reshape(-1, self.n_wavelengths)

        # train on (transformed) total radiance prediction
        # IMPORTANT: make sure no log-transforms are used and proxy transform is coupled to target stats
        outputs = outputs + proxy

        return outputs
    
    
    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if isinstance(self.input_dim_reduction, PCA):
            checkpoint['input_dim_reduction'] = self.input_dim_reduction.pca_dict()
        super().on_save_checkpoint(checkpoint)
        
    def load_model_state(self, checkpoint, map_location=None):

        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction.load_pca_dict(checkpoint['input_dim_reduction'], map_location)
            
        super().load_model_state(checkpoint, map_location=map_location)

    def _apply(self, convert):
        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction._apply(convert)
        
        return super()._apply(convert)
    

class SequenceNNSurrogateRTM(NNSurrogateRTM):
    def __init__(self, 
                 output_dim: int,
                 input_dim_reduction: Optional[Union[DimensionalityReduction, Encoder]] = None,
                 pos_encoding: Optional[torch.nn.Module] = None,
                 proxy_module: Optional[torch.nn.Module] = None,
                 profile_module: Optional[torch.nn.Module] = None,
                 input_processing: Optional[torch.nn.Module] = None,
                 use_proxy_input: bool = False,
                 use_profile_input: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_wavelengths = output_dim
        self.input_dim_reduction = input_dim_reduction
        self.pos_encoding = pos_encoding
        self.input_processing = input_processing

        self.proxy_module = proxy_module
        self.use_proxy_input = use_proxy_input

        self.profile_module = profile_module
        self.use_profile_input = use_profile_input


    def forward(self, inputs, **kwargs) -> torch.Tensor:

        if self.use_proxy_input:
            # proxy targets have shape [batch_size, n_wavelengths]
            # expand inputs and add them as additional channels to proxy spectra
            proxy = kwargs[self.proxy_quantity].unsqueeze(-1)

            if self.proxy_module:
                proxy = self.proxy_module(proxy)

        if isinstance(self.input_dim_reduction, (DimensionalityReduction, Encoder)):
            inputs = self.input_dim_reduction.encode(inputs)

        if self.use_profile_input:
            # profile inputs have shape [batch_size, n_layers, n_vars]
            profile_input = kwargs['profile_inputs']

            if self.profile_module:
                profile_input = self.profile_module(profile_input)
                inputs = torch.cat([inputs, profile_input], dim=1)

        # inputs have shape [batch_size, n_features]
        # expand across wavelengths
        inputs = inputs.unsqueeze(1).expand(-1, self.n_wavelengths, -1)

        if self.use_proxy_input:
            inputs = torch.cat([inputs, proxy], dim=-1)


        if self.input_processing:
            inputs = self.input_processing(inputs)

        if self.pos_encoding:
            # add positional information to inputs
            inputs = self.pos_encoding(inputs)

        
        outputs = self.forward_model(inputs).reshape(-1, self.n_wavelengths)

        return outputs

    
    
    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if isinstance(self.input_dim_reduction, PCA):
            checkpoint['input_dim_reduction'] = self.input_dim_reduction.pca_dict()
        super().on_save_checkpoint(checkpoint)
        
    def load_model_state(self, checkpoint, map_location=None):

        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction.load_pca_dict(checkpoint['input_dim_reduction'], map_location)
            
        super().load_model_state(checkpoint, map_location=map_location)

    def _apply(self, convert):
        if isinstance(self.input_dim_reduction, PCA):
            self.input_dim_reduction._apply(convert)
        
        return super()._apply(convert)



class NonScatteringRTM(SurrogateRTM):
    def __init__(self, proxy_quantity='proxy_targets', **kwargs):
        super().__init__(**kwargs)

        self.proxy_quantity = proxy_quantity

    def forward(self, **input_kwargs) -> torch.Tensor:
        return input_kwargs[self.proxy_quantity]



class BidirectionalLSTMEncoder(torch.nn.Module):
    """
    Encodes a multi-dimensional sequence to a hidden feature vector
    """
    def __init__(self, n_in, n_hidden, n_out, n_layers=1, dropout_p=0.0):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=n_in, 
                                  hidden_size=n_hidden, 
                                  num_layers=n_layers, 
                                  batch_first=True, 
                                  dropout=dropout_p, 
                                  bidirectional=True)
        
        self.fc_out = torch.nn.Linear(2 * n_hidden, n_out)

    def forward(self, sequence):
        
        out, _ = self.lstm(sequence)

        # use concatenation of last forward and backward hidden state to define features
        out = self.fc_out(out[:, -1, :])

        return out
    

class Transformer(torch.nn.Module):
    def __init__(self, num_layers, d_model, d_out, nhead=1, 
                 dim_feedforward=128, dropout_p=0.0):
        super().__init__()
        
        layer = torch.nn.TransformerEncoderLayer(d_model=d_model, 
                                                 nhead=nhead, 
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout_p,
                                                 batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=num_layers)

        self.fc_out = torch.nn.Linear(d_model, d_out)

    def forward(self, x):
        # x has shape (batch_size, seq_length, n_features)
        x = self.transformer(x)
        x = self.fc_out(x)

        return x
    

class ResNet1D(torch.nn.Module):
    def __init__(self, n_channels: List[int], n_out: int = 1, kernel_size: int = 3, stride: int = 1, **kwargs):
        super().__init__()

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_channels = kwargs.get('n_fc_layers', 1)
        self.is_sequence_model = True

        self.layers = nn.ModuleList([
            layers.ResidualBlock1D(n_channels[i], n_channels[i+1], 
                                   kernel_size=kernel_size, stride=stride, 
                                   dropout_p=self.dropout_p)
                                   for i in range(len(n_channels) - 1)
        ])
        self.fc_out = torch.nn.Linear(n_channels[-1], n_out)

    #     self.reset_parameters()

    # def reset_parameters(self):

    #     self.fc_hidden.apply(init_weights)
    #     init_weights(self.fc_in)
    #     init_weights(self.fc_out)

    def forward(self, x):
        # x has shape (batch_size, seq_length, n_channels)
        # conv1d expects shape (batch_size, n_channels, seq_length)
        x = x.permute((0, 2, 1))

        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = x.permute(0, 2, 1)
        x = self.fc_out(x)

        return x

    

