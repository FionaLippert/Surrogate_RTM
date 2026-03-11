####################################################
# Test suite for the RadiativeTransferDataset class
####################################################

import pytest
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np

from rtm import datasets, utils

# fix random seed for reproducibility
seed = 0
pl.seed_everything(seed)

# path to synthetic data
DATADIR = '/nfs/Sentinel5/users/andrewb/RemoTeC/s5L2OP_v3.0.0/RUNDIR/Dynamic_input'

@ pytest.fixture
def make_dataset():
    # setup radiative transfer dataset
    transform = utils.NumpyToTensor()
    target_transform = utils.RTMCompose([utils.NumpyToTensor(), utils.LogTransform()])

    dataset = datasets.RadiativeTransferDataset(
        data_dir=DATADIR,
        transform=transform,
        target_transform=target_transform,
    )
    return dataset

@ pytest.fixture
def normalized_dataset(make_dataset):
    # setup a normalized radiative transfer dataset
    dataset = make_dataset

    input_norm, target_norm = dataset.normalization(dataset.indices)
    dataset.add_transforms(input_norm, target_norm)

    return dataset

def test_input_dimensions(make_dataset):
    # check dimensions of input data
    dataset = make_dataset
    
    assert len(dataset) == dataset.inputs.shape[0]
    assert len(dataset.inputs.shape) == 2
    assert dataset[0]['inputs'].shape == dataset.inputs.shape[1:]
    assert dataset[dataset.indices]['inputs'].shape == dataset.inputs.shape
    assert sum(list(dataset.input_shapes.values())) == dataset.inputs.shape[1]

def test_target_dimensions(make_dataset):
    # check dimensions of target data
    dataset = make_dataset

    assert len(dataset) == dataset.targets.shape[0]
    assert len(dataset.targets.shape) == 2
    assert dataset[0]['targets'].shape == dataset.targets.shape[1:]
    assert dataset[dataset.indices]['targets'].shape == dataset.targets.shape


def test_input_normalization(normalized_dataset):
    # check normalization of dataset
    dataset = normalized_dataset

    # load all inputs via __getitem__
    inputs = dataset[dataset.indices]['inputs']

    idx = 0
    for name, dim in dataset.input_shapes.items():
        # check normalization for each input variable separately
        var_i = inputs[:, idx:idx+dim]

        assert var_i.mean().item() == pytest.approx(0, abs=1e-04)
        assert (var_i.std().item() == pytest.approx(1, abs=1e-04)) or (var_i.std().item() == 0)

        idx += dim

def test_target_normalization(normalized_dataset):
    # check normalization of dataset
    dataset = normalized_dataset

    # load all targets via __getitem__
    targets = dataset[dataset.indices]['targets']

    # check normalization for each wavelength separately
    mean = targets.mean(0)
    std = targets.std(0)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-04)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-04)


def test_inverse_transform(normalized_dataset):
    # check inverse transform of dataset
    dataset = normalized_dataset
    targets = dataset.targets

    # apply transform and invert again
    transformed = dataset.target_transform(targets)
    targets_inv = dataset.target_transform.inverse(transformed)

    # check inverse transform
    assert np.allclose(targets_inv, targets, atol=1e-04)

def test_data_split(make_dataset):
    dataset = make_dataset
  
    train_set, test_set = dataset.get_data_split(test_size=0.2)

    # check if split is the same every time
    for _ in range(3):
        train_set_i, test_set_i = dataset.get_data_split(test_size=0.2)

        for idx in train_set_i.indices:
            assert idx in train_set.indices
        
        for idx in test_set_i.indices:
            assert idx in test_set.indices

