import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, ConstantLR
from typing import Callable, Union, List, Dict, Tuple, Optional
import numpy as np
from hydra.utils import instantiate

from rtm import rtm_transforms, utils, layers
from rtm.dimensionality_reduction import DimensionalityReduction, Encoder, Decoder, PCA
from rtm.utils import init_weights


class SurrogateRTMWrapper(pl.LightningModule):
    def __init__(self, 
                 forward_model: torch.nn.Module,
                 transforms: Optional[rtm_transforms.RTMTransformCollection] = None,
                 eval_scale: Optional[torch.Tensor] = None, 
                 **kwargs):

        super().__init__()

        self.forward_model = forward_model
        self.transforms = transforms
        self.forward_model.set_transforms(transforms)
        
        self.eval_scale = eval_scale

    def set_transforms(self, transforms: rtm_transforms.RTMTransformCollection):
        print('Setting transforms...')
        self.transforms = transforms
        self.forward_model.set_transforms(transforms)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """runs RTM prediction"""
        return self.forward_model.forward(*args, **kwargs)
    
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

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch['targets']
        inputs = batch['inputs']
        proxy_targets = batch.get('proxy_targets', None)
        profile_inputs = batch.get('profile_inputs', None)

        # make prediction (and transform back to physical quantities)
        outputs = self.forward(inputs,
                               proxy_targets=proxy_targets,
                               profile_inputs=profile_inputs,
                               train=False)
        
        # make sure that targets are final radiances
        targets = self.forward_model.combine_with_proxy(targets, proxy_targets)

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
                       },
                       sync_dist=True)

    def predict_step(self, batch, batch_idx):
        """reconstruct input and transform back to physical space"""
        targets = batch['targets']
        inputs = batch['inputs']
        proxy_targets = batch.get('proxy_targets', None)
        profile_inputs = batch.get('profile_inputs', None)

        # make prediction (and transform back to physical quantities)
        outputs = self.forward(inputs,
                               proxy_targets=proxy_targets,
                               profile_inputs=profile_inputs,
                               train=False)
        
        # make sure that targets are final radiances
        targets = self.forward_model.combine_with_proxy(targets, proxy_targets)

        # change to numpy arrays
        outputs, targets = self.prepare_outputs(outputs, targets)

        return outputs, targets
    
    def save_hydra_config(self, config: Dict):
        self.hydra_config = config


class NNSurrogateRTMWrapper(SurrogateRTMWrapper):
    def __init__(self, 
                 lr: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        
        # training hyperparameters
        self.lr = lr
        self.l2_reg = kwargs.get('l2_reg', 0)
        self.loss_scale = kwargs.get('loss_scale', 1.0)

        # list of loss terms making up the overall loss
        self.loss_terms = kwargs.get('loss_terms', [torch.nn.MSELoss()])

        # jacobian regularization
        jacobian_weights = kwargs.get('jacobian_weights', 1.0)
        jacobian_weights = torch.tensor(jacobian_weights, dtype=torch.float32)
        self.register_buffer('jacobian_weights', jacobian_weights)
        self.jacobian_loss_scale = kwargs.get('jacobian_loss_scale', 0.0)
        self.jacobian_regularization = self.jacobian_loss_scale > 0.0

        # learning rate scheduling
        self.lr_schedulers = kwargs.get('lr_schedulers', None)
        self.lr_milestones = kwargs.get('lr_milestones', None)
        self.lr_scheduler_interval = kwargs.get('lr_scheduler_interval', 'epoch')

    
    # def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:

    #     # prepare inputs
    #     if self.input_transform is not None:
    #         inputs = self.input_transform(inputs)

    #     # prepare proxy
    #     if self.proxy_quantity in kwargs and self.proxy_transform is not None:
    #         kwargs[self.proxy_quantity] = self.proxy_transform(kwargs[self.proxy_quantity])

    #     return self.forward_model(inputs, **kwargs)


    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_fct in self.loss_terms:
            loss = loss + loss_fct(outputs, targets)

        # if using SGD without regularization, this is equivalent to scaling the lr
        loss = loss * self.loss_scale

        return loss
    
    def jacobian_loss(self, nn_jacobian: torch.Tensor, target_jacobian: torch.Tensor) -> torch.Tensor: 
        """compute jacobian regularization loss"""
        if target_jacobian is None:
            raise ValueError('Target jacobian required for jacobian regularization')

        # mean squared error between jacobians (weighted by jacobian_weights)
        squared_errors = torch.pow(nn_jacobian - target_jacobian, 2)
        # weight each input dimension separately
        weighted_squared_errors = squared_errors * self.jacobian_weights
        # average over input dimensions, weighted by jacobian_weights
        weighted_mse = torch.sum(weighted_squared_errors, dim=-1) / torch.sum(self.jacobian_weights)
        # average over output dimensions
        jac_loss = torch.mean(weighted_mse)
        # scale jacobian loss
        jac_loss = jac_loss * self.jacobian_loss_scale

        return jac_loss

    def eval_step(self, batch, split) -> torch.Tensor:

        inputs = batch['inputs']
        proxy_targets = batch.get('proxy_targets', None)
        profile_inputs = batch.get('profile_inputs', None)

        # make prediction (without transforming back to physical quantities)
        outputs = self.forward(inputs,
                               proxy_targets=proxy_targets,
                               profile_inputs=profile_inputs,
                               train=True)

        # compute loss function
        targets = self.forward_model.transform_targets(batch['targets'])
        loss = self.compute_loss(outputs, targets)

        # log loss and log-loss
        self.log(f'{split}/loss', jac_loss, on_epoch=True, 
                    on_step=(True if split == 'train' else False),
                    prog_bar=(True if split == 'train' else False), 
                    logger=True, sync_dist=True)
        self.log(f'{split}/log_loss', torch.log(jac_loss), on_epoch=True, 
                    on_step=(True if split == 'train' else False),
                    prog_bar=(True if split == 'train' else False), 
                    logger=True, sync_dist=True)

        # compute jacobian if required for regularization
        if self.jacobian_regularization:
            target_jacobian = batch.get('target_jacobian', None)
            proxy_jacobian = batch.get('proxy_jacobian', None)
            jacobian = self.jacobian_exact(inputs,
                                           proxy_targets=proxy_targets,
                                           profile_inputs=profile_inputs,
                                           proxy_jacobian=proxy_jacobian)
            
            jac_loss = self.jacobian_loss(jacobian, target_jacobian)
            loss = loss + jac_loss

            # log jacobian loss separately
            self.log(f'{split}/jacobian_loss', jac_loss, on_epoch=True, 
                    on_step=(True if split == 'train' else False),
                    prog_bar=(True if split == 'train' else False), 
                    logger=True, sync_dist=True)
            self.log(f'{split}/log_jacobian_loss', torch.log(jac_loss), on_epoch=True, 
                    on_step=(True if split == 'train' else False),
                    prog_bar=(True if split == 'train' else False), 
                    logger=True, sync_dist=True)


        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.eval_step(batch, split='train')
        # self.log('train/loss', loss, on_step=True, on_epoch=True, 
        #          prog_bar=True, logger=True, sync_dist=True)
        # self.log('train/log_loss', torch.log(loss), on_step=True, 
        #          on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
            
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step(batch, split='val')
        # self.log('val/loss', loss, on_epoch=True, logger=True, sync_dist=True)
        # self.log('val/log_loss', torch.log(loss), on_epoch=True, logger=True, sync_dist=True)

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
    

    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if hasattr(self, 'hydra_config'):
            checkpoint['hydra_config'] = self.hydra_config

    
    def on_load_checkpoint(self, checkpoint):
        print(f'Loading model state on device {self.device}...')
        self.load_model_state(checkpoint, map_location=self.device)
        
        
    def load_model_state(self, checkpoint, map_location=None):

        # if transforms are part of checkpoint, make sure model has correct transforms setup
        if 'transforms' in checkpoint['hydra_config']:
            self.set_transforms(instantiate(checkpoint['hydra_config']['transforms']))

        # load standard pytorch state_dict
        self.load_state_dict(checkpoint['state_dict'])

        # # load additional objects from checkpoint
        # self.target_transform = checkpoint['target_transform']
        # self.proxy_transform = checkpoint['proxy_transform']

        # make sure everything is on the same device
        self.to(map_location)




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
        """runs RTM forward model prediction"""
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

        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()

        return outputs, targets

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        outputs = self.forward(**batch)

        # outputs = self.forward(**batch)
        # if self.target_transform:
        #     # undo normalization and transforms
        #     outputs = self.target_transform.inverse(outputs)

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
                       },
                       sync_dist=True)

    def predict_step(self, batch, batch_idx):
        """reconstruct input and transform back to physical space"""
        targets = batch[self.target_quantity]
        # if self.target_transform is not None:
        #     targets = self.target_transform(targets)

        # outputs = self.forward(**batch)
        # if self.target_transform:
        #     # undo normalization and transforms
        #     outputs = self.target_transform.inverse(outputs)

        outputs = self.forward(**batch)
        outputs, targets = self.prepare_outputs(outputs, targets)

        return outputs, targets
    
    def save_hydra_config(self, config: Dict):
        self.hydra_config = config


    
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


    def prepare_inputs(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        # prepare inputs
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        # prepare proxy
        if self.proxy_quantity in kwargs and self.proxy_transform is not None:
            kwargs[self.proxy_quantity] = self.proxy_transform(kwargs[self.proxy_quantity])

        return inputs, kwargs
    

    def forward(self, inputs: torch.Tensor, train=False, **kwargs) -> torch.Tensor:
        # normalize and transform inputs if applicable
        inputs, kwargs_prep = self.prepare_inputs(inputs, **kwargs)
        
        # make prediction
        outputs = self.forward_model(inputs, **kwargs_prep)

        # for final prediction, translate back to physical quantities
        if not train:
            if self.target_transform is not None:
                outputs = self.target_transform.inverse(outputs)
        
        return outputs


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
            print('Transforming targets...')
            targets = self.target_transform(targets)

        # make prediction
        outputs = self.forward(**batch, train=True)

        # compute loss function
        loss = self.compute_loss(outputs, targets)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.eval_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('train/log_loss', torch.log(loss), on_step=True, 
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
            
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step(batch)
        self.log('val/loss', loss, on_epoch=True, logger=True, sync_dist=True)
        self.log('val/log_loss', torch.log(loss), on_epoch=True, logger=True, sync_dist=True)

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
    

    def on_save_checkpoint(self, checkpoint):
        # save additional objects to checkpoint 
        if hasattr(self, 'hydra_config'):
            checkpoint['hydra_config'] = self.hydra_config

    
    def on_load_checkpoint(self, checkpoint):
        print(f'Loading model state on device {self.device}...')
        self.load_model_state(checkpoint, map_location=self.device)
        
        
    def load_model_state(self, checkpoint, map_location=None):

        # if transforms are part of checkpoint, make sure model has correct transforms setup
        if 'transforms' in checkpoint:
            self.transforms = instantiate(checkpoint['hydra_config']['transforms'])

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
    

    def forward(self, inputs: torch.Tensor, train=False, **kwargs) -> torch.Tensor:
        # normalize and transform inputs if applicable
        inputs, kwargs_prep = self.prepare_inputs(inputs, **kwargs)
        
        # make prediction
        outputs = self.forward_model(inputs, **kwargs_prep)

        # for final prediction, translate back to physical quantities
        if not train:
            if self.target_transform is not None:
                outputs = self.target_transform.inverse(outputs)

            # go from difference to total radiance
            proxy = kwargs[self.proxy_quantity]
            outputs = proxy + outputs
        
        return outputs


    # def test_step(self, batch, batch_idx):
    #     """evaluate reconstruction performance"""
    #     # get targets
    #     targets = batch[self.target_quantity]

    #     # make prediction
    #     # outputs = self.forward(**batch)
    #     # if self.target_transform:
    #     #     # undo normalization and transforms
    #     #     outputs = self.target_transform.inverse(outputs)
    #     outputs = self.predict(**batch)
    #     outputs, targets = self.prepare_outputs(outputs, targets)

    #     self.outputs.append(outputs)
    #     self.targets.append(targets)

    # def predict_step(self, batch, batch_idx):
    #     """make prediction"""
    #     # get targets
    #     targets = batch[self.target_quantity]

    #     # make prediction
    #     # outputs = self.forward(**batch)
    #     # if self.target_transform:
    #     #     # undo normalization and transforms
    #     #     outputs = self.target_transform.inverse(outputs)
    #     outputs = self.predict(**batch)
    #     outputs, targets = self.prepare_outputs(outputs, targets)

    #     return outputs, targets
    
    
class NNSurrogateRTMRatio(NNSurrogateRTM):
    """Surrogate model that predicts the ratio between RTM and proxy (non-scattering RTM)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs: torch.Tensor, train=False, **kwargs) -> torch.Tensor:
        # normalize and transform inputs if applicable
        inputs, kwargs_prep = self.prepare_inputs(inputs, **kwargs)
        
        # make prediction
        outputs = self.forward_model(inputs **kwargs_prep)

        # for final prediction, translate back to physical quantities
        if not train:
            if self.target_transform is not None:
                outputs = self.target_transform.inverse(outputs)

            # go from difference to total radiance
            proxy = kwargs[self.proxy_quantity]
            outputs = proxy * outputs
        
        return outputs

    # def test_step(self, batch, batch_idx):
    #     """evaluate reconstruction performance"""
    #     # get targets
    #     targets = batch[self.target_quantity]

    #     # make prediction
    #     # outputs = self.forward(**batch)
    #     # if self.target_transform:
    #     #     # undo normalization and transforms
    #     #     outputs = self.target_transform.inverse(outputs)
    #     outputs = self.predict(**batch)
    #     outputs, targets = self.prepare_outputs(outputs, targets)

    #     self.outputs.append(outputs)
    #     self.targets.append(targets)

    # def predict_step(self, batch, batch_idx):
    #     """make prediction"""
    #     # get targets
    #     targets = batch[self.target_quantity]
        
    #     # make prediction
    #     # outputs = self.forward(**batch)
    #     # if self.target_transform:
    #     #     # undo normalization and transforms
    #     #     outputs = self.target_transform.inverse(outputs)
    #     outputs = self.predict(**batch)
    #     outputs, targets = self.prepare_outputs(outputs, targets)

    #     return outputs, targets

class ForwardModel(torch.nn.Module):
    def __init__(self, 
                 forward_module: torch.nn.Module, 
                 transforms: Optional[rtm_transforms.RTMTransformCollection] = None
                 ):
        super().__init__()

        self.forward_module = forward_module
        self.transforms = transforms

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
        
    def prepare_inputs(self, 
                       inputs: torch.Tensor, 
                       proxy_targets: torch.Tensor = None,
                       ) -> Tuple[torch.Tensor]:
        # prepare inputs
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        # prepare proxy
        if proxy_targets is not None and self.proxy_transform is not None:
            proxy_targets = self.proxy_transform(proxy_targets)

        return inputs, proxy_targets


    # def forward(self, 
    #             inputs: torch.Tensor, 
    #             proxy_targets: torch.Tensor = None,
    #             train: bool = False,
    #             ) -> torch.Tensor:

    #     # prepare inputs
    #     if self.input_transform is not None:
    #         inputs = self.input_transform(inputs)

    #     # TODO: also prepare proxy is applicable

    #     # make prediction
    #     outputs = self.forward(inputs, **kwargs)

    #     # prepare outputs
    #     if self.target_transform is not None:
    #         outputs = self.target_transform.inverse(outputs)

    #     return outputs
    
    # def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
    #     return self.forward_module(inputs, **kwargs)

class NNForwardModel(torch.nn.Module):
    """
    Base class for neural network forward models
    """
    def __init__(self, proxy_operation: Optional[str] = None):
        super().__init__()

        # operation used to combine neural network output with proxy targets
        # one of 'difference', 'ratio', None
        self.proxy_operation = proxy_operation

        print(f'Using proxy operation {self.proxy_operation}...')

        # transforms can be set later based on training dataset
        self.transforms = None

        self.null_transform = rtm_transforms.NullTransform()

    def set_transforms(self, transforms):
        self.transforms = transforms

    @property
    def input_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'pixel_transform', self.null_transform)
    
    @input_transform.setter
    def input_transform(self, input_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'pixel_transform': input_transform})
        else:
            raise Warning('Trying to set input_transform for a model without transforms')

    @property
    def target_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'target_transform', self.null_transform)
    
    @target_transform.setter
    def target_transform(self, target_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'target_transform': target_transform})
        else:
            raise Warning('Trying to set target_transform for a model without transforms')

    @property
    def proxy_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'proxy_transform', self.null_transform)
    
    @proxy_transform.setter
    def proxy_transform(self, proxy_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'proxy_transform': proxy_transform})
        else:
            raise Warning('Trying to set proxy_transform for a model without transforms')
        
    @property
    def profile_transform(self) -> rtm_transforms.RTMTransform:
        return getattr(self.transforms, 'profile_transform', self.null_transform)
    
    @profile_transform.setter
    def profile_transform(self, profile_transform: rtm_transforms.RTMTransform):
        if self.transforms is not None:
            self.transforms.set_transforms({'profile_transform': profile_transform})
        else:
            raise Warning('Trying to set input_transform for a model without transforms')

    def prepare_inputs(self, 
                       inputs: torch.Tensor, 
                    #    proxy_targets: Optional[torch.Tensor] = None,
                    #    profile_inputs: Optional[torch.Tensor] = None,
                       ) -> torch.Tensor:
        # prepare inputs
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        # # prepare proxy
        # if proxy_targets is not None and self.proxy_transform is not None:
        #     proxy_targets = self.proxy_transform(proxy_targets)

        # # prepare profiles
        # if profile_inputs is not None and self.profile_transform is not None:
        #     profile_inputs = self.profile_transform(profile_inputs)

        return inputs
    
    def prepare_proxy(self, proxy_targets: torch.Tensor) -> torch.Tensor:
        # apply proxy transform if applicable
        if self.proxy_transform is not None and proxy_targets is not None:
            proxy_targets = self.proxy_transform(proxy_targets)

        return proxy_targets
    
    def prepare_profiles(self, profile_inputs: torch.Tensor) -> torch.Tensor:
        # apply profile transform if applicable
        if self.profile_transform is not None and profile_inputs is not None:
            profile_inputs = self.profile_transform(profile_inputs)

        return profile_inputs
    
    def transform_targets(self, targets):
        # apply target transform if applicable
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return targets
    
    def combine_with_proxy(self, outputs: torch.Tensor, proxy_targets: torch.Tensor = None):

        # combine NN outputs with proxy targets if applicable
        if self.proxy_operation == 'difference':
            outputs = outputs + proxy_targets
        elif self.proxy_operation == 'ratio':
            outputs = outputs * proxy_targets

        return outputs

    def prepare_outputs(self,
                        outputs: torch.Tensor,
                        proxy_targets: Optional[torch.Tensor] = None,
                        train: Optional[bool] = False,
                        combine_with_proxy: Optional[bool] = True,
                        ) -> torch.Tensor:
        
        if not train:
            # undo normalization and transforms if applicable
            if self.target_transform is not None:
                outputs = self.target_transform.inverse(outputs)

            if combine_with_proxy and proxy_targets is not None:
                outputs = self.combine_with_proxy(outputs, proxy_targets)

        return outputs
    
    def forward(self, 
                inputs: torch.Tensor,
                proxy_targets: Optional[torch.Tensor] = None,
                profile_inputs: Optional[torch.Tensor] = None,
                train: Optional[bool] = False,
                normalized_inputs: Optional[bool] = False,
                combine_with_proxy: Optional[bool] = True,
                ) -> torch.Tensor:
        """Run neural network forward model"""
        raise NotImplementedError
    
    def _predict(self, 
                inputs: torch.Tensor,
                proxy_targets: torch.Tensor, 
                profile_inputs: torch.Tensor,
                normalized_inputs: Optional[bool] = False,
                combine_with_proxy: Optional[bool] = True,
                ) -> torch.Tensor:
        
        # make prediction (in eval mode)
        return self.forward(inputs, 
                            proxy_targets=proxy_targets,
                            profile_inputs=profile_inputs,
                            train=False,
                            normalized_inputs=normalized_inputs,
                            combine_with_proxy=combine_with_proxy)
    
    def jacobian_exact(self, 
                 inputs: torch.Tensor,
                 proxy_targets: Optional[torch.Tensor] = None,
                 profile_inputs: Optional[torch.Tensor] = None,
                 proxy_jacobian: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
        """
        Compute exact jacobian using autograd.
        """
        
        if inputs.dim() == 1:
            # add batch dimension if inputs are 1D
            inputs = inputs.unsqueeze(0)
            if proxy_targets is not None:
                proxy_targets = proxy_targets.unsqueeze(0)
            if profile_inputs is not None:
                profile_inputs = profile_inputs.unsqueeze(0)
            if proxy_jacobian is not None:
                proxy_jacobian = proxy_jacobian.unsqueeze(0)
            dim = 1
        else:
            dim = inputs.dim()
        
        # define batch dimensions
        batch_dims = [0, 0, 0, None, None]
        if proxy_targets is None:
            batch_dims[1] = None
        if profile_inputs is None:
            batch_dims[2] = None
        batch_dims = tuple(batch_dims)
        
        # compute jacobian w.r.t. inputs using autograd,
        # parallelizing over batch dimension
        f_jac = torch.func.jacrev(self._predict, argnums=0)
        f_jac_vmap = torch.func.vmap(f_jac, in_dims=batch_dims, out_dims=0)

        # for ratio or diff models, compute jacobian for correction term only 
        # i.e. without combining it with proxy
        combine_with_proxy = False
        normalized_inputs = False
        jac = f_jac_vmap(inputs, 
                         proxy_targets, 
                         profile_inputs, 
                         normalized_inputs, 
                         combine_with_proxy)

        # if applicable, combine with proxy jacobian
        jac = self.combine_jacobians(jac, proxy_jacobian, inputs, proxy_targets, profile_inputs)

        if dim == 1:
            # remove batch dimension if inputs were 1D
            jac = jac.squeeze(0)

        return jac # shape (batch_size, n_outputs, n_features)
    
    def combine_jacobians(self,
                          jac: torch.Tensor,
                          proxy_jacobian: torch.Tensor,
                          inputs: torch.Tensor,
                          proxy_targets: torch.Tensor,
                          profile_inputs: Optional[torch.Tensor] = None,
                          ) -> torch.Tensor:
        """
        Combine jacobian of the neural network with the jacobian of the proxy targets.
        """
        
        if self.proxy_operation == 'ratio':
            # if proxy_operation is "ratio", apply chain rule to combine jacobians
            assert proxy_jacobian is not None, \
                'Proxy jacobian must be provided for Jacobian computation.'
            assert proxy_targets is not None, \
                'Proxy targets must be provided for Jacobian computation.'

            # first, get correction factor (without combining with proxy)
            correction_factor = self._predict(inputs, 
                                              proxy_targets, 
                                              profile_inputs, 
                                              normalized_inputs=False,
                                              combine_with_proxy=False)
            # apply chain rule to combine jacobians
            # jac = correction_factor * proxy_jacobian + proxy_targets * jac_correction
            # correction_factor has shape (batch_size, n_outputs)
            # proxy_jacobian has shape (batch_size, n_outputs, n_features)
            # jac has shape (batch_size, n_outputs, n_features)
            jac = correction_factor.unsqueeze(-1) * proxy_jacobian + proxy_targets.unsqueeze(-1) * jac

        elif self.proxy_operation == 'difference':
            # if proxy operation is "difference", we can just add the two jacobians

            assert proxy_jacobian is not None, \
                'Proxy jacobian must be provided for Jacobian computation.'
            
            jac = jac + proxy_jacobian

        return jac

    def jacobian_approx(self,
                        inputs: torch.Tensor,
                        proxy_targets: Optional[torch.Tensor] = None,
                        profile_inputs: Optional[torch.Tensor] = None,
                        proxy_jacobian: Optional[torch.Tensor] = None,
                        feature_indices: Optional[List[int]] = None,
                        eps: Optional[Union[float, List[float]]] = 1e-6,
                        rel_eps: Optional[bool] = False,
                        central_difference: Optional[bool] = False,
                        ) -> torch.Tensor:
        """
        Compute jacobian using finite differences.
        """

        if inputs.dim() == 1:
            # add batch dimension if inputs are 1D
            inputs = inputs.unsqueeze(0)
            if proxy_targets is not None:
                proxy_targets = proxy_targets.unsqueeze(0)
            if profile_inputs is not None:
                profile_inputs = profile_inputs.unsqueeze(0)
            if proxy_jacobian is not None:
                proxy_jacobian = proxy_jacobian.unsqueeze(0)
            dim = 1
        else:
            dim = inputs.dim()

        # forward pass (for correction term only, without combining with proxy)
        base_output = self._predict(inputs, 
                                    proxy_targets, 
                                    profile_inputs, 
                                    normalized_inputs=False, 
                                    combine_with_proxy=False)
        n_outputs = base_output.size(1)
        batch_size = inputs.size(0)
        n_features = inputs.size(1)

        # initialize jacobian
        jac = torch.zeros(batch_size, n_outputs, n_features, dtype=inputs.dtype, device=inputs.device)
        # jac = jac.type_as(inputs, device=inputs.device)

        if feature_indices is None:
            feature_indices = range(n_features)
        if not isinstance(eps, list):
            eps = [eps] * len(feature_indices)

        if rel_eps:
            eps = [eps_i * torch.abs(inputs[:, i]).mean() for i, eps_i in zip(feature_indices, eps)]

        # compute jacobian using finite differences
        for i, eps_i in zip(feature_indices, eps):
            # perturb input
            h = torch.zeros_like(inputs)
            h[:, i] = eps_i

            # forward pass with perturbed input
            perturbed_output_fwd = self._predict(inputs + h, 
                                             proxy_targets, 
                                             profile_inputs,
                                             normalized_inputs=False,
                                             combine_with_proxy=False)

            if central_difference:
                perturbed_output_bwd = self._predict(inputs - h,
                                                 proxy_targets,
                                                 profile_inputs,
                                                 normalized_inputs=False,
                                                 combine_with_proxy=False)
                jac[:, :, i] = 0.5 * (perturbed_output_fwd - perturbed_output_bwd) / eps_i
            else:
                jac[:, :, i] = (perturbed_output_fwd - base_output) / eps_i

        # if applicable, combine with proxy jacobian
        jac = self.combine_jacobians(jac, proxy_jacobian, inputs, proxy_targets, profile_inputs)

        if dim == 1:
            # remove batch dimension if inputs were 1D
            jac = jac.squeeze(0)

        # jac has shape (batch_size, n_outputs, n_features)
        return jac

    

class LatentNNForwardModel(NNForwardModel):
    """
    Simple forward model that first encodes to the input features to a 
    lower-dimensional latent space (e.g. using PCA or autoencoder),
    then passes them through an MLP, and finally applies a decoder 
    to obtain the final output.
    """
    def __init__(self,  
                 core_nn: torch.nn.Module, 
                 proxy_nn: Optional[torch.nn.Module] = None,
                 encoder: Optional[Encoder] = None, 
                 decoder: Optional[Decoder] = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.core_nn = core_nn

        # optional proxy processing (e.g. non-scattering radiances)
        self.proxy_nn = proxy_nn

        # optional encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, 
                inputs: torch.Tensor,
                proxy_targets: Optional[torch.Tensor] = None,
                profile_inputs: Optional[torch.Tensor] = None,
                train: Optional[bool] = False,
                normalized_inputs: Optional[bool] = False,
                combine_with_proxy: Optional[bool] = True,
                ) -> torch.Tensor:
        # predicts radiance spectrum from input features, and optionally proxy spectra

        # apply transforms and normalization if applicable
        if not normalized_inputs:
            inputs = self.prepare_inputs(inputs)

        proxy_transf = self.prepare_proxy(proxy_targets)
        profile_transf = self.prepare_profiles(profile_inputs)
        
        if self.encoder is not None:
            # print(f'Input stats: {inputs.mean(dim=0)}, {inputs.std(dim=0)}')
            inputs = self.encoder.encode(inputs)

        if self.proxy_nn is not None:
            proxy_transf = self.proxy_nn(proxy_transf)
            inputs = torch.cat([inputs, proxy_transf], dim=-1)

        outputs = self.core_nn(inputs)

        if self.decoder is not None:
            outputs = self.decoder.decode(outputs)

        # undo any transforms and normalization, 
        # and combine with proxy targets if applicable
        # proxy_targets = kwargs.get('proxy_targets', None)
        outputs = self.prepare_outputs(outputs, proxy_targets, train, combine_with_proxy)

        return outputs
    


class SequenceNNForwardModel(NNForwardModel):
    def __init__(self, 
                 sequence_length: int,
                 sequence_module: torch.nn.Module,
                 encoder: Optional[Encoder] = None,
                 pos_encoding: Optional[torch.nn.Module] = None,
                 proxy_module: Optional[torch.nn.Module] = None,
                 profile_module: Optional[torch.nn.Module] = None,
                 input_processing: Optional[torch.nn.Module] = None,
                 use_proxy_input: bool = False,
                 use_profile_input: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length

        self.sequence_module = sequence_module
        self.encoder = encoder
        self.pos_encoding = pos_encoding
        self.input_processing = input_processing

        self.proxy_module = proxy_module
        self.use_proxy_input = use_proxy_input

        self.profile_module = profile_module
        self.use_profile_input = use_profile_input


    def forward(self, 
                inputs: torch.Tensor,
                proxy_targets: Optional[torch.Tensor] = None,
                profile_inputs: Optional[torch.Tensor] = None,
                train: Optional[bool] = False,
                normalized_inputs: Optional[bool] = False,
                combine_with_proxy: Optional[bool] = True,
                ) -> torch.Tensor:
        
        if inputs.dim() == 1:
            # single input, expand to include batch dimension
            inputs = inputs.unsqueeze(0)
            if proxy_targets is not None:
                proxy_targets = proxy_targets.unsqueeze(0)
            if profile_inputs is not None:
                profile_inputs = profile_inputs.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        
        # apply transforms and normalization if applicable
        if not normalized_inputs:
            inputs = self.prepare_inputs(inputs)

        proxy_transf = self.prepare_proxy(proxy_targets)
        profile_transf = self.prepare_profiles(profile_inputs)

        # inputs, transformed_kwargs = self.prepare_inputs(inputs, 
        #                                                  proxy_targets=proxy_targets, 
        #                                                  profile_inputs=profile_inputs)

        if self.encoder is not None:
            inputs = self.encoder.encode(inputs)

        if self.use_profile_input:
            # profile inputs have shape [batch_size, n_layers, n_vars]
            profile_inputs = profile_transf

            if self.profile_module is not None:
                profile_inputs = self.profile_module(profile_inputs)

            # make sure profile features are flattened
            profile_inputs = profile_inputs.reshape(inputs.size(0), -1)
            
            # add profile information to inputs
            inputs = torch.cat([inputs, profile_inputs], dim=1)

        # inputs have shape [batch_size, n_features] --> expand across wavelengths
        inputs = inputs.unsqueeze(1).expand(-1, self.sequence_length, -1)

        if self.use_proxy_input:
            # proxy targets have shape [batch_size, n_wavelengths]
            # expand inputs and add them as additional channels to proxy spectra
            proxy = proxy_transf.unsqueeze(-1)
            # proxy = proxy_targets.unsqueeze(-1)

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
        outputs = self.sequence_module(inputs)
        outputs = outputs.reshape(-1, self.sequence_length)

        # undo any normalization and transforms,
        # and combine with proxy targets if applicable
        # proxy_targets = kwargs.get('proxy_targets', None)
        outputs = self.prepare_outputs(outputs, proxy_targets, train, combine_with_proxy)

        # if compute_jac:
        #     # compute jacobian of the forward model
        #     jac_f = torch.func.jacrev(forward_nn, argnums=0)(test_inputs, test_proxy)

        if single_input:
            outputs = outputs.squeeze(0)

        return outputs
    
    # def compute_jac(self, 
    #                 inputs: torch.Tensor,
    #                 proxy_targets: Optional[torch.Tensor] = None,
    #                 profile_inputs: Optional[torch.Tensor] = None,
    #                 ) -> torch.Tensor:
    #     # compute jacobian of the forward model



        
  

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
    def __init__(self, n_channels: List[int], n_out: int = 1, 
                 kernel_size: int = 3, stride: int = 1, **kwargs):
        super().__init__()

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_channels = kwargs.get('n_fc_layers', 1)
        self.is_sequence_model = True

        nonlinearity = kwargs.get('nonlinearity', 'relu')

        self.layers = nn.ModuleList([
            layers.ResidualBlock1D(n_channels[i], n_channels[i+1], 
                                   kernel_size=kernel_size, stride=stride,
                                   nonlinearity=nonlinearity, 
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

    

