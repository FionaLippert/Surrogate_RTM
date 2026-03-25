import torch
from torch.utils.data import Dataset
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from typing import Any, List, Optional
import os

from rtm import rtm_transforms

def prepare_transforms(cfg_transforms: DictConfig, 
                       dataset: Dataset, 
                       indices: List[int], 
                       **kwargs):

    # initialize transforms
    transforms = instantiate(cfg_transforms)

    # compute normalization of inputs and outputs
    normalization = dataset.normalization(
        transforms=transforms, 
        indices=indices,
        **kwargs
    )
    transforms.add_transforms(normalization)

    transforms.add_transform_to_all(rtm_transforms.DoubleToFloat())

    return transforms


def prepare_train_dataset(cfg_dataset: DictConfig):

    # get training data
    dataset = instantiate(cfg_dataset, train=True)

    # split training data into training and validation sets
    train_indices, val_indices = dataset.get_data_split(cfg_dataset.val_size)

    return dataset, train_indices, val_indices

def prepare_test_dataset(cfg_dataset: DictConfig):

    # get test data
    dataset = instantiate(cfg_dataset, train=False)

    return dataset


def load_train_dataset(cfg_dataset: OmegaConf, 
                       data_path: Optional[str] = None):

    artifact_name = f'{cfg_dataset.name}_train_val'
    if data_path is not None:
        data_path = os.path.join(data_path, artifact_name)

    if data_path is not None and os.path.exists(data_path):
        # if it exists, load test data from disk
        print(f'Loading training data from {data_path}...')
        dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'), weights_only=False)
        train_indices = torch.load(os.path.join(data_path, 'train_indices.pt'), weights_only=False)
        val_indices = torch.load(os.path.join(data_path, 'val_indices.pt'), weights_only=False)
    else:
        # if not, load and normalize dataset from scratch
        print(f'Training data could not be loaded from {data_path}!')
        print(f'Preparing dataset from scratch...')

        dataset, train_indices, val_indices = prepare_train_dataset(cfg_dataset)

    return dataset, train_indices, val_indices


def load_test_dataset(cfg_dataset: OmegaConf, 
                      data_path: Optional[str] = None, 
                      run = None):
    
    artifact_name = f'{cfg_dataset.name}_test'
    if data_path is not None:
        data_path = os.path.join(data_path, artifact_name)

    # if exists, load test data from disk, if not, try to load from W&B,
    # otherwise create from scratch
    if data_path is not None and os.path.exists(data_path):
        # if exists, load test data from disk
        print(f'Loading test data from {data_path}...')
        dataset = torch.load(os.path.join(data_path, 'test_dataset.pt'), weights_only=False)
    else:
        print(f'Test data could not be loaded from {data_path}!')
        try:
            # if possible, load dataset from wandb project
            dataset_artifact = run.use_artifact(f'{artifact_name}:latest')
            dataset_path = dataset_artifact.download(root=data_path)

            print(f'Downloaded data to {dataset_path}')

            dataset = torch.load(os.path.join(dataset_path, 'test_dataset.pt'), weights_only=False)

            print(f'Test data successfully loaded from project {wandb.run.project}')
        except:
            # if not, load and normalize dataset from scratch
            print(f'Test data could not be loaded from project {wandb.run.project}')
            print(f'Preparing dataset from scratch...')
            
            dataset = prepare_test_dataset(cfg_dataset) #, transforms=transforms)

    return dataset


def prepare_data_and_log(cfg_dataset: DictConfig,
                         data_path: str,
                         wandb_project: Optional[str] = None,
                         transforms: Optional[List[rtm_transforms.RTMTransform]] = None):
    # load training and test data, and log everything as artifacts to W&B

    with wandb.init(project=wandb_project, job_type='prepare-data') as run:

        if cfg_dataset.test_size < 1.0:
            # prepare training data
            train_dataset, train_indices, val_indices = prepare_train_dataset(cfg_dataset)
            train_data_path = os.path.join(data_path, f'{cfg_dataset.name}_train_val')
            os.makedirs(train_data_path, exist_ok=True)
        
            print('Logging training data to W&B...')
            train_data_artifact = wandb.Artifact(
                        f'{cfg_dataset.name}_train_val', type="dataset",
                        description="Training dataset, can be further split into training and validation sets",
                        metadata={"source": cfg_dataset._target_,
                                "config": OmegaConf.to_container(cfg_dataset, resolve=True)}
            )

            fn_train_dataset = os.path.join(train_data_path, "train_dataset.pt")
            torch.save(train_dataset, fn_train_dataset)
            train_data_artifact.add_file(fn_train_dataset)

            fn_train_indices = os.path.join(train_data_path, "train_indices.pt")
            torch.save(train_indices, fn_train_indices)
            train_data_artifact.add_file(fn_train_indices)

            fn_val_indices = os.path.join(train_data_path, "val_indices.pt")
            torch.save(train_indices, fn_val_indices)
            train_data_artifact.add_file(fn_val_indices)

            wandb.run.log_artifact(train_data_artifact)

        if cfg_dataset.test_size > 0.0:
            # prepare test data
            test_dataset = prepare_test_dataset(cfg_dataset)

            print('Logging test data to W&B...')
            test_data_artifact = wandb.Artifact(
                        f'{cfg_dataset.name}_test', type="dataset",
                        description="Test dataset",
                        metadata={"source": cfg_dataset._target_,
                                "config": OmegaConf.to_container(cfg_dataset, resolve=True)}
                )

            test_data_path = os.path.join(data_path, f'{cfg_dataset.name}_test')
            os.makedirs(test_data_path, exist_ok=True)

            fn_test_dataset = os.path.join(test_data_path, "test_dataset.pt")
            torch.save(test_dataset, fn_test_dataset)
            test_data_artifact.add_file(fn_test_dataset)

            wandb.run.log_artifact(test_data_artifact)


def prepare_data_and_save(cfg_dataset: DictConfig, 
                          data_path: str):
    # load training and test data, and save everything to disk

    if cfg_dataset.test_size < 1.0:
        # prepare training data
        train_dataset, train_indices, val_indices = prepare_train_dataset(cfg_dataset)
    
        print('Saving training data...')

        artifact_name = f'{cfg_dataset.name}_train_val'
        dataset_path = os.path.join(data_path, artifact_name)
        os.makedirs(dataset_path, exist_ok=True)

        # log config
        with open(os.path.join(dataset_path, 'config.yaml'), 'w') as file:
            OmegaConf.save(cfg_dataset, file, resolve=True)

        # save training data
        torch.save(train_dataset, os.path.join(dataset_path, 'train_dataset.pt'),
                   pickle_protocol=4)

        # save training and validation indices
        torch.save(train_indices, os.path.join(dataset_path, 'train_indices.pt'),
                   pickle_protocol=4)
        torch.save(val_indices, os.path.join(dataset_path, 'val_indices.pt'),
                   pickle_protocol=4)
        
        # log config
        with open(os.path.join(dataset_path, 'config.yaml'), 'w') as file:
            OmegaConf.save(cfg_dataset, file, resolve=True)

    if cfg_dataset.test_size > 0.0:
        # prepare test data
        test_dataset = prepare_test_dataset(cfg_dataset)

        artifact_name = f'{cfg_dataset.name}_test'
        dataset_path = os.path.join(data_path, artifact_name)
        os.makedirs(dataset_path, exist_ok=True)

        print('Saving test data...')
        torch.save(test_dataset, os.path.join(dataset_path, 'test_dataset.pt'),
                   pickle_protocol=4)

        # log config
        with open(os.path.join(dataset_path, 'config.yaml'), 'w') as file:
            OmegaConf.save(cfg_dataset, file, resolve=True)

