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
    dataset.add_transforms(*[utils.DoubleToFloat()]*4)

    return dataset, train_indices, val_indices

def prepare_test_dataset(dataset_cfg: DictConfig, 
                      transforms: Optional[List[utils.RTMTransform]] = None):

    # get test data
    dataset = instantiate(dataset_cfg, train=False)

    # apply transforms (if provided)
    if transforms: dataset.set_transforms(*transforms)

    return dataset

def load_train_dataset_from_wandb(run, dataset_cfg: OmegaConf, data_path: Optional[str] = None):

    artifact_name = f'{dataset_cfg.name}_train_val:latest'

    try:
        artifact_name = f'{dataset_cfg.name}_train_val'
        dataset_path = os.path.join(data_path, artifact_name)
        # if possible, load dataset from disk, otherwise load from wandb
        if not os.path.isdir(dataset_path):
            dataset_artifact = run.use_artifact(f'{artifact_name}:latest')
            dataset_path = dataset_artifact.download(root=dataset_path)
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


def load_test_dataset_from_wandb(run, 
                                 dataset_cfg: OmegaConf, 
                                 data_path: Optional[str] = None, 
                                 transforms: Optional[List[utils.RTMTransform]] = None):

    # TODO: also allow for simply loading the dataset from disk
    try:
        artifact_name = f'{dataset_cfg.name}_test'
        dataset_path = os.path.join(data_path, artifact_name)
        # if possible, load dataset from disk, otherwise load from wandb
        if not os.path.isdir(dataset_path):
            dataset_artifact = run.use_artifact(f'{artifact_name}:latest')
            dataset_path = dataset_artifact.download(root=dataset_path)
            print(f'Downloaded data from W&B to {dataset_path}')

        dataset = torch.load(os.path.join(dataset_path, 'test_dataset.pt'), weights_only=False)

        if transforms is not None:
            dataset.set_transforms(*transforms)

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
                         data_path: str,
                         wandb_project: Optional[str] = None,
                         transforms: Optional[List[utils.RTMTransform]] = None):
    # load training and test data, as well as transforms, and log everything as artifacts to wandb

    with wandb.init(project=wandb_project, job_type='prepare-data') as run:

        if dataset_cfg.test_size < 1.0:
            # prepare training data
            train_dataset, train_indices, val_indices = prepare_train_dataset(dataset_cfg)
            transforms = train_dataset.get_transforms()
            print(train_dataset.target_transform)
            print(train_dataset.proxy_transform)
            train_data_path = os.path.join(data_path, f'{dataset_cfg.name}_train_val')
            os.makedirs(train_data_path, exist_ok=True)
        
            print('Logging training data to W&B...')
            train_data_artifact = wandb.Artifact(
                        f'{dataset_cfg.name}_train_val', type="dataset",
                        description="Training dataset, can be further split into training and validation sets",
                        metadata={"source": dataset_cfg._target_,
                                "config": OmegaConf.to_container(dataset_cfg, resolve=True)}
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

        
            # with train_data_artifact.new_file("train_dataset.pt", mode="wb") as file:
            #     torch.save(train_dataset, file)
            # with train_data_artifact.new_file("train_indices.pt", mode="wb") as file:
            #     torch.save(train_indices, file)
            # with train_data_artifact.new_file("val_indices.pt", mode="wb") as file:
            #     torch.save(val_indices, file)

            wandb.run.log_artifact(train_data_artifact)

            print('Logging normalization to W&B...')
            normalization = wandb.Artifact(
                    f'{dataset_cfg.name}_transforms', type="normalization",
                    description="Transforms applied to the dataset"
            )
            normalization_path = os.path.join(data_path, f'{dataset_cfg.name}_transforms')
            os.makedirs(normalization_path, exist_ok=True)

            for name, transf in zip(['pixel', 'profile', 'target', 'proxy'], transforms):
                fn_transf = os.path.join(normalization_path, f"{name}_transform.pt")
                torch.save(transf, fn_transf)
                normalization.add_file(fn_val_indices)
                # with normalization.new_file(f"{name}_transform.pt", mode="wb") as file:
                    # torch.save(transf, file)

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
            # with test_data_artifact.new_file("test_dataset.pt", mode="wb") as file:
                # torch.save(test_dataset, file)

            test_data_path = os.path.join(data_path, f'{dataset_cfg.name}_test')
            os.makedirs(test_data_path, exist_ok=True)

            fn_test_dataset = os.path.join(test_data_path, "test_dataset.pt")
            torch.save(test_dataset, fn_test_dataset)
            test_data_artifact.add_file(fn_test_dataset)

            wandb.run.log_artifact(test_data_artifact)

