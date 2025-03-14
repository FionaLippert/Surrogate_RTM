import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from typing import Union, List, Dict, Tuple, Callable, Optional

from rtm import utils
from rtm.utils import init_weights


class Embedding(nn.Embedding):
    """
    nn.Embedding with custom initialization
    """
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 padding_idx: Optional[int] = None, 
                 **kwargs):
        super().__init__(num_embeddings, 
                         embedding_dim, 
                         padding_idx=padding_idx, 
                         **kwargs)
        nn.init.normal_(self.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(self.weight[padding_idx], 0)


class Linear(nn.Linear):
    """
    nn.Linear with custom initialization
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 **kwargs):
        super().__init__(in_features, 
                         out_features, 
                         bias=bias, 
                         **kwargs)
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0.0)

# class MLP(torch.nn.Module):
#     def __init__(self, n_in, n_hidden, n_out, **kwargs):
#         super(MLP, self).__init__()

#         self.dropout_p = kwargs.get('dropout_p', 0)
#         self.n_layers = kwargs.get('n_fc_layers', 1)

#         self.fc_in = torch.nn.Linear(n_in, n_hidden)
#         self.fc_hidden = nn.ModuleList([torch.nn.Linear(n_hidden, n_hidden)
#                                         for _ in range(self.n_layers - 1)])
#         self.fc_out = torch.nn.Linear(n_hidden, n_out)

#         self.reset_parameters()

#     def reset_parameters(self):

#         self.fc_hidden.apply(init_weights)
#         init_weights(self.fc_in)
#         init_weights(self.fc_out)

#     def forward(self, features):

#         x = F.relu(self.fc_in(features))
#         x = F.dropout(x, p=self.dropout_p, training=self.training)

#         for l in self.fc_hidden:
#             x = F.relu(l(x))
#             x = F.dropout(x, p=self.dropout_p, training=self.training)

#         x = self.fc_out(x)

#         return x
    

class MLP(torch.nn.Module):
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
    

# class MLP(torch.nn.Module):
#     """
#     Simple MLP layer with varying layer widths

#     Args:
#     hidden_dims: list of hidden dimensions (i.e. layer widths)
#     dropout_p: dropout probability
#     """
#     def __init__(self,
#                  hidden_dims: List[int],
#                  dropout_p: float = 0.0,
#                  nonlinearity: Callable = F.relu,
#                  **kwargs
#                  ):
#         super().__init__()

#         self.fc_layers = torch.nn.ModuleList(
#             [torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)]
#             )

#         self.dropout_p = dropout_p
#         self.nonlinearity = nonlinearity

#         # initialize weights
#         self.reset_parameters()

#     def reset_parameters(self):
#         for layer in self.fc_layers:
#             init_weights(layer)
    
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         for layer in self.fc_layers[:-1]:
#             z = self.nonlinearity(layer(z))
#             z = F.dropout(z, p=self.dropout_p, training=self.training)
#         # no activation on last layer
#         x = self.fc_layers[-1](z)
#         return x



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

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

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
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # define encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [torch.nn.Linear(enc_dims[i], enc_dims[i+1]) for i in range(len(enc_dims) - 1)]
            )
        self._latent_dim = enc_dims[-1]

        # if > 0, add noise to input data during training
        self.noise_level = noise_level

        self.dropout_p = dropout_p

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
        for layer in self.encoder_layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        # no activation on last layer
        x = self.encoder_layers[-1](x)
        return x
    
class PCAEncoder(Encoder):
    """
    Simple encoder using pre-defined PCA components

    Args:
    n_components: number of PCA components
    fixed_components: if true, keep PCA components fixed during training, otherwise allow to adjust
    """
    def __init__(self,
                 n_components: int, 
                 fixed_components: bool = True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._latent_dim = n_components
        self.fixed_components = fixed_components
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        self.pca.fit(data)

        components = torch.tensor(self.pca.components_, dtype=torch.float32)
        mean = torch.tensor(self.pca.mean_, dtype=torch.float32)

        if self.fixed_components:
            # keep PCA transform as is
            self.register_buffer('components', components) # shape (n_components, features)
            self.register_buffer('mean', mean) # shape (features,)
        else:
            # alternatively, allow to adjust transformation during training
            self.components = torch.nn.Parameter(components, requires_grad=True)
            self.mean = torch.nn.Parameter(mean, requires_grad=True)

    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.components, - self.mean @ self.components.T)
        return z
    

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

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

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
                 **kwargs
                 ):
        super().__init__()

        self.decoder_layers = torch.nn.ModuleList(
            [torch.nn.Linear(dec_dims[i], dec_dims[i+1]) for i in range(len(dec_dims) - 1)]
            )
        self._latent_dim = dec_dims[0]

        self.dropout_p = dropout_p

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.decoder_layers:
            init_weights(layer)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers[:-1]:
            z = F.relu(layer(z))
            z = F.dropout(z, p=self.dropout_p, training=self.training)
        # no activation on last layer
        x = self.decoder_layers[-1](z)
        return x
    

class PCADecoder(Decoder):
    """
    Simple decoder using pre-defined PCA components

    Args:
    n_components: number of PCA components
    fixed_components: if true, keep PCA components fixed during training, otherwise allow to adjust
    """
    def __init__(self,
                 n_components: int, 
                 fixed_components: bool = True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self._latent_dim = n_components
        self.fixed_components = fixed_components
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        self.pca.fit(data)

        components = torch.tensor(self.pca.components_, dtype=torch.float32)
        mean = torch.tensor(self.pca.mean_, dtype=torch.float32)

        if self.fixed_components:
            # keep PCA transform as is
            self.register_buffer('components', components) # shape (n_components, features)
            self.register_buffer('mean', mean) # shape (features,)
        else:
            # alternatively, allow to adjust transformation during training
            self.components = torch.nn.Parameter(components, requires_grad=True)
            self.mean = torch.nn.Parameter(mean, requires_grad=True)

    def explained_variance_ratio(self):
        return self.pca.explained_variance_ratio_
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.linear(z, self.components.T, self.mean)
        return x


##################################################
# Positional Encoding for Transformer-based models
##################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_wavelengths, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encodings once
        position = torch.arange(num_wavelengths, dtype=torch.float)  # Shape: [num_wavelengths]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Calculate the positional encodings
        encoding = torch.zeros(num_wavelengths, d_model)
        encoding[:, 0::2] = torch.sin(position.view(-1, 1) * div_term)
        encoding[:, 1::2] = torch.cos(position.view(-1, 1) * div_term)
        self.register_buffer('pos_encoding', encoding.unsqueeze(0)) # Shape: [1, num_wavelengths, d_model]

    def forward(self, x):

        return self.dropout(x + self.pos_encoding)
    

##################################################
# 1D Residual Block for ResNet-based models
##################################################

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_p=0.0):
        super().__init__()

        padding = (kernel_size - 1) // 2  # To maintain the same sequence length after the convolution
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # TODO: identify correct position to place dropout within the residual block
        # self.droupout = nn.Dropout(p=dropout_p)
        
        if stride > 1 or in_channels != out_channels:
            # reduce channels and/or sequence length
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)
        return out