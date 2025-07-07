import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn import decomposition
import torch_pca
import numpy as np
from typing import Union, List, Dict, Tuple, Callable

from rtm import utils, layers
from rtm.utils import init_weights

class DimensionalityReduction(pl.LightningModule):
    def __init__(self, 
                 target_quantity: str = 'targets',
                 target_transform: utils.RTMTransform = None, 
                 eval_scale: torch.Tensor = None,
                 **kwargs):
        super().__init__()

        self._target_transform = target_transform
        self.eval_scale = eval_scale

        # quantity to encode/decode
        self.target_quantity = target_quantity

        self._latent_dim = None
        self._input_dim = None

    @property
    def target_transform(self) -> utils.RTMTransform:
        return self._target_transform
    
    @target_transform.setter
    def target_transform(self, target_transform: utils.RTMTransform) -> None:
        self._target_transform = target_transform

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
    
    @property
    def input_dim(self) -> int:
        return self._input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """runs encoding and returns encoded input"""
        raise NotImplementedError
    
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        """runs encoding and decoding and returns reconstructed input"""
        raise NotImplementedError
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode input data"""
        raise NotImplementedError
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """decode latent representation"""
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        """training step for neural network models"""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """validation step for neural network models"""
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        """evaluate reconstruction performance"""
        data = batch[self.target_quantity]
        reconstruction = self.encode_decode(data)

        data = data.detach().cpu()
        reconstruction = reconstruction.detach().cpu()

        if self.target_transform:
            # undo normalization and transforms
            data = self.target_transform.inverse(data)
            reconstruction = self.target_transform.inverse(reconstruction)
        else:
            data = data.numpy()
            reconstruction = reconstruction.numpy()

        self.reconstruction.append(reconstruction)
        self.data.append(data)
    
    def on_test_epoch_start(self) -> None:
        """initialize lists to store predictions and targets"""
        self.reconstruction = []
        self.data = []
    
    def on_test_epoch_end(self) -> None:

        # process all predictions and targets from the test set
        data = np.concatenate(self.data, axis=0)
        reconstruction = np.concatenate(self.reconstruction, axis=0)
       
        # compute evaluation metrics, average across input/output dimension
        rmse = utils.rmse(reconstruction, data, axis=0).mean()
        nrmse = utils.nrmse(reconstruction, data, z=self.eval_scale, axis=0).mean()
        mape = utils.mape(reconstruction, data, axis=0).mean()
        bias = utils.bias(reconstruction, data, axis=0).mean()
        r2 = utils.r2_score(reconstruction, data, axis=0).mean()

        test_dict = {
            'test/RMSE': rmse,
            'test/NRMSE': nrmse,
            'test/MAPE': mape,
            'test/bias': bias,
            'test/R2': r2,
        }

        if self.target_quantity == 'targets':
            # radiance-specific metrics
            mape_continuum = utils.mape_continuum(reconstruction, data, axis=0).mean()
            smape = utils.smape(reconstruction, data, axis=0).mean()
            rel_bias = utils.relative_bias(reconstruction, data, axis=0).mean()

            # compute R2 on log-transformed data
            # ensure that outputs are positive
            reconstruction = np.maximum(reconstruction, np.finfo(reconstruction.dtype).eps)
            r2_log = utils.r2_score(np.log(reconstruction), np.log(data), axis=0).mean()

            test_dict.update({
                'test/MAPE_continuum': mape_continuum,
                'test/SMAPE': smape,
                'test/rel_bias': rel_bias,
                'test/R2_log': r2_log,
            })

        self.log_dict(test_dict)
        

    def predict_step(self, batch, batch_idx):
        """reconstruct input and transform back to physical space"""
        data = batch[self.target_quantity]
        reconstruction = self.encode_decode(data)
        
        data = data.detach().cpu()
        reconstruction = reconstruction.detach().cpu()

        if self.target_transform:
            # undo normalization and transforms
            data = self.target_transform.inverse(data)
            reconstruction = self.target_transform.inverse(reconstruction)
        else:
            data = data.numpy()
            reconstruction = reconstruction.numpy()

        return reconstruction, data
    

class PCA(DimensionalityReduction):
    def __init__(self, n_components, **kwargs):
        super().__init__(**kwargs)

        self.pca = torch_pca.PCA(n_components=n_components)

        self._latent_dim = n_components
        self._input_dim = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return z
    
    def encode_decode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pca.transform(x.view(-1, self.input_dim))
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.pca.inverse_transform(z.view(-1, self.latent_dim))
        return x_hat
    
    def fit(self, x: torch.Tensor):
        self.pca.fit_transform(x)
        self._input_dim = x.shape[-1]

    def explained_variance_ratio(self) -> torch.Tensor:
        return self.pca.explained_variance_ratio_.sum().item()
    
    def pca_dict(self):
        pca_dict = {
            'pca': self.pca.__dict__,
            '_input_dim': self._input_dim,
            '_latent_dim': self._latent_dim
        }
        return pca_dict
    
    def load_pca_dict(self, state_dict, map_location=None):
        for key, value in state_dict['pca'].items():
            if isinstance(value, torch.Tensor):
                value = value.to(map_location)
            setattr(self.pca, key, value)
        self._input_dim = state_dict['_input_dim']
        self._latent_dim = state_dict['_latent_dim']

    def _apply(self, convert):
        newself = super()._apply(convert)
        device = newself.device

        print(f'Move PCA to device {device}')
        for key, value in newself.pca.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(newself.pca, key, value.to(device))

        return newself


##################################################
# Encoder modules
##################################################

class Encoder(nn.Module):
    """
    Encoder base class
    """
    def __init__(self, **kwargs):
        super().__init__()

        self._latent_dim = None
        self._input_dim = None

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
    
    @property
    def input_dim(self) -> int:
        return self._input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
   

class MLPEncoder(Encoder):
    """
    Simple encoder with fully-connected (FC) layers

    Args:
    enc_dims: list of dimensions for encoder layers
    noise_level: std of Gaussian noise added to input during training
    dropout_p: dropout probability
    """
    def __init__(self,
                 enc_dims: List[int], 
                 noise_level: float = 0.0,
                 dropout_p: float = 0.0,
                 bias: bool = True,
                 nonlinearity: str = 'relu',
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # define encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [torch.nn.Linear(enc_dims[i], enc_dims[i+1], bias=bias) for i in range(len(enc_dims) - 1)]
            )
        self._latent_dim = enc_dims[-1]
        self._input_dim = enc_dims[0]

        # if > 0, add noise to input data during training
        self.noise_level = noise_level

        self.dropout_p = dropout_p

        self.nonlinearity = getattr(F, nonlinearity)

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.encoder_layers:
            init_weights(layer)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.noise_level * torch.randn_like(x).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # add Gaussian noise to input
            x = self.add_noise(x)
        z = self.encode(x)
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.encoder_layers[:-1]):
            x = layer(x)
            # if i > 0:
            x = self.nonlinearity(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        # no activation on last layer
        x = self.encoder_layers[-1](x)
        return x

    
class ResNet1DEncoder(Encoder):
    def __init__(self, input_channels, hidden_dim=64, latent_dim=256, num_blocks=4, stride=1, **kwargs):
        super().__init__(**kwargs)

        self._latent_dim = latent_dim

        self.verbose = kwargs.get('verbose', False)

        # Initial convolutional layer to reduce the input channels and sequence length
        self.initial_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=7, stride=stride, padding=3)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)

        # Create a sequence of residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = hidden_dim

        for _ in range(num_blocks):
            self.res_blocks.append(layers.ResidualBlock1D(in_channels, in_channels * 2, stride=stride))
            in_channels *= 2

        # Global average pooling gets rid of the remaining sequence length dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(in_channels, latent_dim)

    def forward(self, x):
        if self.verbose: print(f'Input shape: {x.shape}')

        x = self.initial_conv(x.unsqueeze(1))

        if self.verbose: print(f'Output shape after initial conv: {x.shape}')

        x = F.relu(self.initial_bn(x))
        if self.verbose: print(f'Output shape after initial batch norm: {x.shape}')

        for block in self.res_blocks:
            x = block(x)
        if self.verbose: print(f'Output shape after residual blocks: {x.shape}')

        x = self.global_pool(x).squeeze(-1)  # Global average pooling to get latent representation
        if self.verbose: print(f'Output shape after global pooling: {x.shape}')

        latent = self.fc(x)
        if self.verbose: print(f'Latent shape: {latent.shape}')

        return latent

    
class RTCNNEncoder(Encoder):
    def __init__(self, 
                 latent_dim: int, 
                 num_wavelengths: int,
                 kernel_size: int = 3, 
                 num_layers: int = 1, 
                 dropout: float = 0.0,
                 verbose: bool = False):
        super().__init__()

        self._latent_dim = latent_dim
        self._input_dim = num_wavelengths
        self.num_wavelengths = num_wavelengths
        self.verbose = verbose

        # determine sequence length of the latent state
        self.latent_length = max(1, num_wavelengths // (2 ** num_layers))

        # Linear adjustment to ensure that sequence length is divisible by 2^num_layers
        self.initial_length = self.latent_length * (2 ** num_layers)
        self.adjustment_layer = nn.Linear(num_wavelengths, self.initial_length)

        # Create a sequence of deconvolutional layers to upsample the sequence
        conv_layers = []
        in_channels = 1
        current_length = self.initial_length

        self.sequence_lengths = [num_wavelengths, current_length]
        self.channels = [in_channels]

        for _ in range(num_layers):
            # channels are doubled every layer
            out_channels = in_channels #* 2
            conv_layers.append(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
            current_length = max(1, current_length // 2)  # halfing the length with each convolution step
            self.sequence_lengths.append(current_length)
            self.channels.append(in_channels)

        self.conv_layers = nn.Sequential(*conv_layers)

        self.latent_projection = nn.Linear(out_channels * self.latent_length, latent_dim)
        self.channels.append(latent_dim)        

    def encode(self, targets):
        # targets: [batch_size, num_wavelengths]
        batch_size = targets.size(0)

        if self.verbose: print(f'Input shape: {targets.shape}')

        # make sure sequence length is multiple of 2^num_layers
        projected = self.adjustment_layer(targets).unsqueeze(1)  # Shape: [batch_size, 1, initial_length]

        if self.verbose: print(f'Adjusted shape: {projected.shape}')

        # convolve the sequence using the convolutional layers
        output = self.conv_layers(projected)  # Shape: [batch_size, num_layers, latent_length]

        if self.verbose: print(f'Output shape after convolutions: {output.shape}')

        # ensure latent state has the required dimensions
        output = self.latent_projection(output.reshape(batch_size, -1))  # Shape: [batch_size, latent_dim]

        if self.verbose: print(f'Latent shape: {output.shape}')

        return output 


class RTBiGRUEncoder(Encoder):
    def __init__(self, 
                 latent_dim: int, 
                 num_wavelengths: int, 
                 d_model: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_wavelengths = num_wavelengths
        self._latent_dim = latent_dim
        
        # Bidirectional GRU for the encoding process
        self.gru = nn.GRU(input_size=1, hidden_size=d_model, num_layers=num_layers, 
                          dropout=dropout, batch_first=True, bidirectional=True)
        
        # Linear layer to map the GRU output to latent state
        # Output size should match the number of directions (2 * d_model for bidirectional)
        self.output_layer = nn.Linear(d_model * 2, latent_dim)

    def encode(self, input_sequence):
        # Pass the input sequence through the Bidirectional GRU
        gru_output, _ = self.gru(input_sequence.unsqueeze(-1))  # gru_output shape: [batch_size, num_wavelengths, 2 * d_model]

        # Map the GRU output to the final radiance spectrum
        latent_state = self.output_layer(gru_output[:, 0])  # Shape: [batch_size, latent_dim]

        return latent_state  # Shape: [batch_size, latent_dim]

 


    

##################################################
# Decoder modules
##################################################

class Decoder(nn.Module):
    """
    Decoder base class
    """
    def __init__(self, **kwargs):
        super().__init__()

        self._latent_dim = None
        self._output_dim = None

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decode(z)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLPDecoder(Decoder):
    """
    Simple decoder with fully-connected (FC) layers

    Args:
    dec_dims: list of dimensions for decoder layers
    dropout_p: dropout probability
    """
    def __init__(self,
                 dec_dims: List[int],
                 dropout_p: float = 0.0,
                 nonlinearity: str = 'relu',
                 output_activation: str = None,
                 **kwargs
                 ):
        super().__init__()

        self.decoder_layers = torch.nn.ModuleList(
            [torch.nn.Linear(dec_dims[i], dec_dims[i+1]) for i in range(len(dec_dims) - 1)]
            )
        self._latent_dim = dec_dims[0]
        self._output_dim = dec_dims[-1]

        self.dropout_p = dropout_p

        self.nonlinearity = getattr(F, nonlinearity)

        if output_activation is not None:
            # add output activation function
            if hasattr(F, output_activation):
                self.output_activation = getattr(F, output_activation)
            elif hasattr(torch, output_activation):
                self.output_activation = getattr(torch, output_activation)
            else:
                raise Warning(f'{output_activation} not found.')

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.decoder_layers:
            init_weights(layer)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers[:-1]:
            z = self.nonlinearity(layer(z))
            z = F.dropout(z, p=self.dropout_p, training=self.training)
        # no activation on last layer
        x = self.decoder_layers[-1](z)

        if self.output_activation:
            x = self.output_activation(x)
        return x
    
class RTCNNDecoder(Decoder):
    def __init__(self, 
                 latent_dim: int, 
                 num_wavelengths: int, 
                 kernel_size: int = 3, 
                 num_layers: int = 1, 
                 dropout: float = 0.0,
                 verbose: bool = False):
        super().__init__()

        self._latent_dim = latent_dim
        self._output_dim = num_wavelengths
        self.num_wavelengths = num_wavelengths
        self.num_filters = 1 #2 ** num_layers
        self.verbose = verbose

        # Initial linear layer to expand the latent state to a starting sequence length
        self.initial_length = max(4, num_wavelengths // (2 ** num_layers))  # Ensure a reasonable starting length
        self.latent_projection = nn.Linear(latent_dim, self.num_filters * self.initial_length)

        # Create a sequence of deconvolutional layers to upsample the sequence
        deconv_layers = []
        in_channels = self.num_filters
        current_length = self.initial_length

        self.sequence_lengths = [current_length]

        for _ in range(num_layers):
            out_channels = max(in_channels // 2, 1)  # Ensure that the number of channels does not go below 1
            deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                )
            )
            deconv_layers.append(nn.ReLU())
            deconv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
            current_length *= 2  # Doubling the length with each deconvolution step
            self.sequence_lengths.append(current_length)

        self.deconv_layers = nn.Sequential(*deconv_layers)

        # Final convolutional layer to map to a single channel
        self.final_layer = nn.Conv1d(in_channels, 1, kernel_size=1)

        # Linear adjustment to ensure the exact sequence length
        self.adjustment_layer = nn.Linear(current_length, num_wavelengths)

        self.sequence_lengths.append(num_wavelengths)

    def decode(self, latent_state):

        if self.verbose: print(f'Latent state shape: {latent_state.shape}')

        # Project the latent state into an initial sequence with multiple channels
        batch_size = latent_state.size(0)
        projected = self.latent_projection(latent_state)  # Shape: [batch_size, num_filters * initial_length]

        if self.verbose: print(f'Projected shape: {projected.shape}')

        # projected = projected.view(batch_size, -1, projected.size(-1) // (2 ** len(self.deconv_layers))) 
        projected = projected.view(batch_size, self.num_filters, self.initial_length)  # Shape: [batch_size, num_filters, initial_length]

        if self.verbose: print(f'Projected shape: {projected.shape}')

        # Upsample the sequence using the deconvolutional layers
        output = self.deconv_layers(projected) 

        if self.verbose: print(f'Output shape after convolutions: {output.shape}')

        output = self.final_layer(output).squeeze(1)  # Shape: [batch_size, 1, sequence_length]

        if self.verbose: print(f'Output shape: {output.shape}')

        # Final 1D convolution to get a different channels and adjust via a linear size reshape to match num wavelengths exactly.
        output = self.adjustment_layer(output)

        if self.verbose: print(f'Output shape after adjustment: {output.shape}')

        return output  
    

class ResNet1DDecoder(Decoder):
    def __init__(self, latent_dim, num_wavelengths, output_channels=1, hidden_dim=64, num_blocks=4, **kwargs):
        super().__init__()

        self._latent_dim = latent_dim
        self.num_wavelengths = num_wavelengths

        self.verbose = kwargs.get('verbose', False)
        
        self.fc = nn.Linear(latent_dim, hidden_dim * (2 ** num_blocks))
        self.res_blocks = nn.ModuleList()
        in_channels = hidden_dim * (2 ** num_blocks)

        self.scale_factors = []
        current_length = 1

        for _ in range(num_blocks):
            self.res_blocks.append(
                layers.ResidualBlock1D(in_channels, in_channels // 2)
            )
            in_channels //= 2
            
            if current_length * 2 < num_wavelengths:
                self.scale_factors.append(2)
                current_length *= 2
            else:
                self.scale_factors.append(1)

        self.final_conv = nn.Conv1d(hidden_dim, output_channels, kernel_size=7, stride=1, padding=3)

    def forward(self, latent):
        if self.verbose: print(f'Latent state shape: {latent.shape}')

        x = self.fc(latent).unsqueeze(-1)  # Shape: [batch_size, hidden_dim, 1]
        if self.verbose: print(f'Output shape after initial linear layer: {x.shape}')

        for block, factor in zip(self.res_blocks, self.scale_factors):
            x = F.interpolate(x, scale_factor=factor, mode='linear', align_corners=False)
            x = block(x)
        if self.verbose: print(f'Output shape after residual blocks: {x.shape}')

        # ensure the output has the correct sequence length
        x = F.interpolate(x, size=self.num_wavelengths, mode='linear', align_corners=False)
        if self.verbose: print(f'Output shape after final interpolation: {x.shape}')

        output = self.final_conv(x).squeeze(1)
        if self.verbose: print(f'Output shape after final convolution: {output.shape}')

        return output



class RTBiGRUDecoder(Decoder):
    def __init__(self, 
                 latent_dim: int, 
                 num_wavelengths: int, 
                 d_model: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_wavelengths = num_wavelengths
        self._latent_dim = latent_dim
        self._output_dim = num_wavelengths
        
        # Bidirectional GRU for the decoding process
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=num_layers, 
                          dropout=dropout, batch_first=True, bidirectional=True)
        
        # Linear layer to map the latent state to the GRU input dimensions
        self.latent_projection = nn.Linear(latent_dim, d_model)
        
        # Linear layer to map the GRU output to radiance values
        # Output size should match the number of directions (2 * d_model for bidirectional)
        self.output_layer = nn.Linear(d_model * 2, 1)

    def decode(self, latent_state):
        # Project the latent state to match the GRU input dimensions
        memory = self.latent_projection(latent_state)  # Shape: [batch_size, d_model]
        memory = memory.unsqueeze(1)  # Shape: [batch_size, 1, d_model]
        
        # Prepare input for the GRU
        # Create a tensor of shape [batch_size, num_wavelengths, d_model]
        # Repeat the projected latent state across the number of wavelengths
        gru_input = memory.repeat(1, self.num_wavelengths, 1)  # Shape: [batch_size, num_wavelengths, d_model]

        # Pass through the Bidirectional GRU
        gru_output, _ = self.gru(gru_input)  # gru_output shape: [batch_size, num_wavelengths, 2 * d_model]

        # Map the GRU output to the final radiance spectrum
        radiance_output = self.output_layer(gru_output)  # Shape: [batch_size, num_wavelengths, 1]

        return radiance_output.squeeze(-1)  # Shape: [batch_size, num_wavelengths]


class RTTransformerDecoder(Decoder):
    def __init__(self, 
                 latent_dim: int, 
                 num_wavelengths, 
                 d_model: int, 
                 nhead: int, 
                 num_decoder_layers: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()

        self.d_model = d_model
        self._latent_dim = latent_dim
        self._output_dim = num_wavelengths
        self.num_wavelengths = num_wavelengths


        # Initialize the positional encoding for wavelengths
        self.positional_encoding = layers.PositionalEncoding(d_model, num_wavelengths)

        # Initialize transformer decoder layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_decoder_layers)

        # Linear layer to map the decoder output to radiance values
        self.output_layer = nn.Linear(d_model, 1)

        # Linear layer to project the latent state to match the decoder input dimensions
        self.latent_projection = nn.Linear(latent_dim, d_model)

        # Predefine the target wavelengths (not needed as an embedding)
        self.register_buffer('target_wavelengths', torch.arange(num_wavelengths, dtype=torch.float32))  # Shape: [num_wavelengths]


    def decode(self, latent_state: torch.Tensor) -> torch.Tensor:

        # Apply positional encoding to the wavelengths
        wavelength_embeds = self.positional_encoding(self.target_wavelengths.view(1, -1, 1))  # Shape: [1, num_wavelengths, d_model]
        wavelength_embeds = wavelength_embeds.expand(latent_state.size(0), -1, self.d_model)  # Shape: [batch_size, num_wavelengths, d_model]

        # Project the latent state to match the decoder input dimensions
        memory = self.latent_projection(latent_state)  # Shape: [batch_size, d_model]
        memory = memory.view(-1, 1, self.d_model)  # Shape: [batch_size, 1, d_model]

        # Pass through the Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=wavelength_embeds,  # Shape: [batch_size, num_wavelengths, d_model]
            memory=memory  # Shape: [batch_size, 1, d_model]
        )

        # Map the decoder output to the final radiance spectrum
        radiance_output = self.output_layer(decoder_output).view(latent_state.size(0), self.num_wavelengths)  # Shape: [batch_size, num_wavelengths]

        return radiance_output  # Shape: [batch_size, num_wavelengths]



class ConvDecoder(Decoder):
    """
    1d convolutional decoder

    Args:
    dec_dims: list of dimensions for decoder layers
    dropout_p: dropout probability
    """
    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 channel_dims: List[int],
                 stride: int = 1,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 padding: int = 1,
                 dropout_p: float = 0.0,
                 **kwargs
                 ):
        super().__init__()

        # input embedding
        self.input_embedding = torch.nn.Linear(latent_dim, channel_dims[0])

        # define conv layers
        self.conv_layers = torch.nn.ModuleList(
            # d_out = (d_in - 1) * stride - 2 * padding + dilation * (kernel_size − 1)
            [torch.nn.ConvTranspose1d(in_channels=channel_dims[i], 
                                      out_channels=channel_dims[i+1], 
                                      kernel_size=kernel_size, 
                                      stride=stride,
                                      dilation=dilation,
                                      padding=padding) for i in range(len(channel_dims) - 1)]
            )
        
        # compute input dimension
        dim = output_dim
        for layer in self.conv_layers:
            dim = dim - dilation * (kernel_size - 1) + 2
            dim = dim + 2 * padding
            dim = dim // stride
            dim = dim + 1

        # TODO: finish this!


##################################################
# Autoencoder models
##################################################


class AE(DimensionalityReduction):
    def __init__(self,
                 output_dim: int,
                 encoder: Encoder, 
                 decoder: Decoder, 
                 lr: float = 0.01, 
                 **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        assert self.encoder.latent_dim == self.decoder.latent_dim
        self._latent_dim = self.encoder.latent_dim

        self.output_dim = output_dim

        self.lr = lr
        self.l2_reg = kwargs.get('l2_reg', 0)
        self.loss_scale = kwargs.get('loss_scale', 1.0)
        self.w_orthogonality = kwargs.get('w_orthogonality', 0.0)

        self.register_buffer('id_input', torch.eye(self.output_dim))
        self.register_buffer('id_latent', torch.eye(self.latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def eval_step(self, batch):
        data = batch[self.target_quantity]
        reconstruction = self.forward(data)
        loss = F.mse_loss(reconstruction, data)

        loss = loss * self.loss_scale

        if self.w_orthogonality:
            # encourage orthogonality of encoder and decoder weights
            orth_loss = self.orthogonality_regularizer()
            loss += self.w_orthogonality * orth_loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.eval_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/log-loss', torch.log(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
            
    def validation_step(self, batch, batch_idx):
        loss = self.eval_step(batch)
        self.log('val/loss', loss, on_epoch=True, logger=True)
        self.log('val/log-loss', torch.log(loss), on_epoch=True, logger=True)

    def orthogonality_regularizer(self) -> torch.Tensor:
        f_enc = self.encoder.encode(self.id_input) # shape [input_dim, latent_dim]
        f_dec = self.decoder.decode(self.id_latent) # shape [latent_dim, input_dim]
        return F.mse_loss(f_dec @ f_enc, self.id_latent)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)


class VAE(DimensionalityReduction):
    def __init__(self, lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)

        self.encoder = layers.MLPEncoder(**kwargs)
        self.decoder = layers.MLPDecoder(**kwargs)

        # encoder latent state contains both mean and log-variance
        assert self.encoder.latent_dim == self.decoder.latent_dim * 2
        self._latent_dim = self.decoder.latent_dim

        # log scale for reconstruction loss
        self.log_scale = Parameter(torch.tensor(kwargs.get('log_scale', 0.0)), requires_grad=False)

        # beta parameter for KL divergence
        self.beta = kwargs.get('beta', 1.0)

        self.lr = lr

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean, log_var = self.encode(x)
        if sample:
            z = self.sample(mean, log_var)
        else:
            z = mean
        x_hat = self.decode(z)
        return x_hat
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
   
        # split into mean and logvar
        mean = z[:, :self.latent_dim]
        logvar = z[:, -self.latent_dim:]
        return mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def sample(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)

        return z
    
    def loss_function(self, 
                      x: torch.Tensor, 
                      x_hat: torch.Tensor,
                      mean: torch.Tensor, 
                      logvar: torch.Tensor) -> torch.Tensor:
        rec_loss = self.rec_loss(x, x_hat)
        kl_div = self.kl_divergence(mean, logvar)
        loss = rec_loss + self.beta * kl_div

        # to be consistent with loss_function_prob, divide by batch_size
        return loss 
    
    def rec_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        rec_loss = F.mse_loss(x_hat, x, reduction='mean')
        return rec_loss

    def kl_divergence(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()
        return kl
    
    # def gaussian_log_likelihood(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #     # if logscale = 0, computes MSE loss
    #     scale = torch.exp(self.log_scale)
    #     dist = torch.distributions.Normal(x_hat, scale)

    #     # measure prob of seeing x
    #     log_pxz = dist.log_prob(x) # also includes constant term, in contast to F.mse_loss
    #     # return log_pxz.sum(dim=-1)
    #     return log_pxz.mean(dim=-1)
    
    # def kl_divergence_prob(self, 
    #                        z: torch.Tensor, 
    #                        mean: torch.Tensor, 
    #                        log_var: torch.Tensor) -> torch.Tensor:
    #     # define prior distribution
    #     p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(log_var))

    #     # define variational distribution
    #     std = torch.exp(0.5 * log_var)
    #     q = torch.distributions.Normal(mean, std)

    #     # get log probabilities of sample z
    #     log_qzx = q.log_prob(z)
    #     log_pz = p.log_prob(z)

    #     # compute kl
    #     kl = (log_qzx - log_pz).sum(-1)

    #     return kl
    
    def eval_step(self, batch):
        x = batch[self.quantity]

        # forward pass
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        x_hat = self.decode(z)

        # # log likelihood
        # ll = self.gaussian_log_likelihood(x_hat, x).mean(0)

        # # kl divergence
        # kl = self.kl_divergence_prob(z, mean, log_var).mean(0)

        # # elbo and loss
        # elbo = ll - self.beta * kl
        # loss = -elbo
        # rec_loss = -ll

        rec_loss = self.rec_loss(x, x_hat)
        kl = self.kl_divergence(mean, log_var)
        loss = rec_loss + self.beta * kl
        elbo = -loss

        return {
            'elbo': elbo,
            'rec_loss': rec_loss,
            'kl': kl,
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        eval_dict = self.eval_step(batch)
        self.log_dict({f'train/{k}': v.detach() for k, v in eval_dict.items()}, 
                      on_step=True, on_epoch=True, logger=True)
 
        return eval_dict['train/loss']
            
    def validation_step(self, batch, batch_idx):
        eval_dict = self.eval_step(batch)
        self.log_dict({f'val/{k}': v.detach() for k, v in eval_dict.items()}, 
                      on_epoch=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



