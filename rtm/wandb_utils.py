import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from typing import Any, List, Optional
import os

from rtm import utils


def prepare_train_dataset(dataset_cfg: DictConfig):

    # get training data
    dataset = instantiate(dataset_cfg, train=True)

    # split training data into training and validation sets
    train_indices, val_indices = dataset.get_data_split(dataset_cfg.val_size)

    # apply normalization based on training set statistics
    dataset.add_normalization(train_indices, **dataset_cfg.normalization)
    dataset.add_transforms(*[utils.DoubleToFloat()]*3)

    return dataset, train_indices, val_indices

def prepare_test_dataset(dataset_cfg: DictConfig, 
                      transforms: Optional[List[utils.RTMTransform]] = None):

    # get test data
    dataset = instantiate(dataset_cfg, train=False)

    # apply transforms (if provided)
    if transforms: dataset.set_transforms(*transforms)

    return dataset

def load_train_dataset(dataset_cfg: OmegaConf, 
                       data_path: Optional[str] = None, 
                       run = None):

    artifact_name = f'{dataset_cfg.name}_train_val'
    if data_path is not None:
        data_path = os.path.join(data_path, artifact_name)

    # if exists, load test data from disk, if not, try to load from W&B,
    # otherwise create from scratch
    if data_path is not None and os.path.exists(data_path):
        print(f'Loading training data from {data_path}...')
        dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'), weights_only=False)
        train_indices = torch.load(os.path.join(data_path, 'train_indices.pt'), weights_only=False)
        val_indices = torch.load(os.path.join(data_path, 'val_indices.pt'), weights_only=False)
    else:
        try:
            # if possible, load dataset from wandb project
            dataset_artifact = run.use_artifact(f'{artifact_name}:latest')
            dataset_path = dataset_artifact.download(root=data_path)

            print(f'Downloaded data to {dataset_path}')

            dataset = torch.load(os.path.join(dataset_path, 'train_dataset.pt'), weights_only=False)
            train_indices = torch.load(os.path.join(dataset_path, 'train_indices.pt'), weights_only=False)
            val_indices = torch.load(os.path.join(dataset_path, 'val_indices.pt'), weights_only=False)

            print(f'Training data successfully loaded from project {wandb.run.project}')
        except:
            # if not, load and normalize dataset from scratch
            print(f'Training data could not be loaded from project {wandb.run.project}')
            print(f'Preparing dataset from scratch...')

            dataset, train_indices, val_indices = prepare_train_dataset(dataset_cfg)

    return dataset, train_indices, val_indices


def load_test_dataset(dataset_cfg: OmegaConf, 
                      data_path: Optional[str] = None, 
                      transforms: Optional[List[utils.RTMTransform]] = None,
                      run = None):
    
    artifact_name = f'{dataset_cfg.name}_test'
    if data_path is not None:
        data_path = os.path.join(data_path, artifact_name)

    # if exists, load test data from disk, if not, try to load from W&B,
    # otherwise create from scratch
    if data_path is not None and os.path.exists(data_path):
        print(f'Loading test data from {data_path}...')
        dataset = torch.load(os.path.join(data_path, 'test_dataset.pt'), weights_only=False)
    else:
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

            if transforms is None:
                # first determine transforms based on training data
                print(f'Loading training data to determine transforms...')
                train_dataset, _, _ = prepare_train_dataset(dataset_cfg)
                transforms = train_dataset.get_transforms()

            print(f'Preparing dataset from scratch...')
            
            dataset = prepare_test_dataset(dataset_cfg, transforms=transforms)

    return dataset


def prepare_data_and_log(dataset_cfg: DictConfig, 
                      wandb_project: Optional[str] = None,
                      transforms: Optional[List[utils.RTMTransform]] = None):
    # load training and test data, as well as transforms, and log everything as artifacts to wandb

    with wandb.init(project=wandb_project, job_type='prepare-data') as run:

        if dataset_cfg.test_size < 1.0:
            # prepare training data
            train_dataset, train_indices, val_indices = prepare_train_dataset(dataset_cfg)
            transforms = train_dataset.get_transforms()
        
            print('Logging training data to W&B...')
            train_data_artifact = wandb.Artifact(
                        f'{dataset_cfg.name}_train_val', type="dataset",
                        description="Training dataset, can be further split into training and validation sets",
                        metadata={"source": dataset_cfg._target_,
                                "config": OmegaConf.to_container(dataset_cfg, resolve=True)}
                )
            with train_data_artifact.new_file("train_dataset.pt", mode="wb") as file:
                torch.save(train_dataset, file)
            with train_data_artifact.new_file("train_indices.pt", mode="wb") as file:
                torch.save(train_indices, file)
            with train_data_artifact.new_file("val_indices.pt", mode="wb") as file:
                torch.save(val_indices, file)
            wandb.run.log_artifact(train_data_artifact)

            print('Logging normalization to W&B...')
            normalization = wandb.Artifact(
                    f'{dataset_cfg.name}_transforms', type="normalization",
                    description="Transforms applied to the dataset"
            )
            for name, transf in zip(['pixel', 'profile', 'target', 'proxy'], transforms):
                with normalization.new_file(f"{name}_transform.pt", mode="wb") as file:
                    torch.save(transf, file)
            wandb.run.log_artifact(normalization)

        elif transforms is None:
            # use all data for testing, but no transforms provided
            print('No transforms provided for independent test set! Please set transforms manually before testing.')

        if dataset_cfg.test_size > 0.0:
            # prepare test data
            test_dataset = prepare_test_dataset(dataset_cfg, transforms=transforms)

            print('Logging test data to W&B...')
            test_data_artifact = wandb.Artifact(
                        f'{dataset_cfg.name}_test', type="dataset",
                        description="Test dataset",
                        metadata={"source": dataset_cfg._target_,
                                "config": OmegaConf.to_container(dataset_cfg, resolve=True)}
                )
            with test_data_artifact.new_file("test_dataset.pt", mode="wb") as file:
                torch.save(test_dataset, file)
            wandb.run.log_artifact(test_data_artifact)


def prepare_data_and_save(dataset_cfg: DictConfig, 
                          data_path: str,
                          transforms: Optional[List[utils.RTMTransform]] = None):
    # load training and test data, as well as transforms, and save everything to disk

    # convert OmegaConf to dict
    # cfg = OmegaConf.to_container(dataset_cfg, resolve=True)

    if dataset_cfg.test_size < 1.0:
        # prepare training data
        train_dataset, train_indices, val_indices = prepare_train_dataset(dataset_cfg)
        transforms = train_dataset.get_transforms()
    
        print('Saving training data...')

        artifact_name = f'{dataset_cfg.name}_train_val'
        dataset_path = os.path.join(data_path, artifact_name)
        os.makedirs(dataset_path, exist_ok=True)

        # log config
        with open(os.path.join(dataset_path, 'config.yaml'), 'w') as file:
            OmegaConf.save(dataset_cfg, file, resolve=True)

        # save training data
        torch.save(train_dataset, os.path.join(dataset_path, 'train_dataset.pt'),
                   pickle_protocol=4)

        # save training and validation indices
        torch.save(train_indices, os.path.join(dataset_path, 'train_indices.pt'),
                   pickle_protocol=4)
        torch.save(val_indices, os.path.join(dataset_path, 'val_indices.pt'),
                   pickle_protocol=4)

        print('Saving normalization...')

        normalization_path = os.path.join(data_path, f'{dataset_cfg.name}_transforms')
        os.makedirs(normalization_path, exist_ok=True)

        # save transforms for all dataset components
        for name, transf in zip(['pixel', 'profile', 'target', 'proxy'], transforms):
            file = os.path.join(normalization_path, f"{name}_transform.pt")
            torch.save(transf, file, pickle_protocol=4)

        # log config
        with open(os.path.join(dataset_path, 'config.yaml'), 'w') as file:
            OmegaConf.save(dataset_cfg, file, resolve=True)


    elif transforms is None:
        # use all data for testing, but no transforms provided
        print('No transforms provided for independent test set! Please set transforms manually before testing.')

    if dataset_cfg.test_size > 0.0:
        # prepare test data
        test_dataset = prepare_test_dataset(dataset_cfg, transforms=transforms)

        artifact_name = f'{dataset_cfg.name}_test'
        dataset_path = os.path.join(data_path, artifact_name)
        os.makedirs(dataset_path, exist_ok=True)

        print('Saving test data...')
        torch.save(test_dataset, os.path.join(dataset_path, 'test_dataset.pt'),
                   pickle_protocol=4)

        # log config
        with open(os.path.join(dataset_path, 'config.yaml'), 'w') as file:
            OmegaConf.save(dataset_cfg, file, resolve=True)

