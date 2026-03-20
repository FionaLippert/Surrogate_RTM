# Surrogate Radiative Transfer Model (RTM)
This code base provides tools to develop and evaluate neural network (NN) based forward models that emulate physics-based forward models for the Sentinel-5 spectrometer.
The aim is to use these emulators to accelerate trace gas and aerosol retrievals. Separate NNs are trained for the NIR, SWIR1 and SWIR3 band, allowing each network to focus on the spectral features relevant to its specific wavelength range.

## Getting started

First, create the provided conda environment and activate it:
```
conda env create -f rtm_env.yml
conda activate rtm
```

### Test Pytorch GPU compatibility

To check if Pytorch can see your GPU, run
```
python -uc "import torch; print('GPU available?', torch.cuda.is_available())"
```

### Known issues

On the Snellius cluster from Surfsara, I had issues installing pytorch directly with conda. 
If this is the case for you, try creating the rtm environment by running
```
conda create --name rtm python=3.13
```
and then installing pytorch (with the cuda version matching your system) via pip:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Afterwards, you can install all the other packages by running:
```
conda env update --file rtm_env_no_pytorch.yml
```

### Hydra config
This code base makes use of [hydra](https://hydra.cc/) to create a hierarchical configuration which can be composed dynamically and allows for overrides through the command line. Have a look at the `scripts/conf` folder to get familiar with the structure of config files.

You can, for example, easily switch between models, by simply adding `model=<model_name>` to your command line (where `<model_name>` is one of the config file names provided in `scripts/conf/model`) when running the `train_eval.py` script. Similarly, you can switch between model training and evaluation, by choosing `task=train`, `task=train+eval`, or `task=eval`.

### Code structure

The code base is structured as follows:

- `rtm`: core implementations of datasets, models, and data transforms, etc.
- `scripts`: python, bash and slurm scripts used for data preparation, model training and evaluation.
    - `scripts/conf`: configuration files used by Hydra 
    - `scripts/sweeps`: W&B configuration files defining hyperparameter sweeps (e.g. over different number of layers) 
    - `test`: test suite using `pytest` (currently not maintained)

## Dataset generation

The data preparation scripts assume that you have simulated a dataset of synthetic radiance measurements using RemoTAP, which is stored in the directory `<data_dir>/synth` with file names `<prefix>_<chunk>_instr1_noise2.nc`, `<prefix>_<chunk>_aux.nc`, `<prefix>_<chunk>_truth.nc`, and optionally `<prefix>_ns_<chunk>_instr1_noise2.nc` containing non-scattering simulations.

To process this data into the format required by the dataset class implemented in `rtm/datasets/uvns_datasts.py`, change to the `scripts` directory and run
```
bash prepare_remotap_data.sh <data_dir> <prefix> <nchunks>
```
where <nchunks> is the number of data chunks that should be included in the training/test dataset. This will split the simulation data by spectral band and create subdirectories `<data_dir>/NIR_preprocessed`, `<data_dir>/SWIR1_preprocessed`, and `<data_dir>/SWIR3_preprocessed`, containing the preprocessed data.

To generate Pytorch training/test datasets from this data, run
```
python prepare_data.py dataset=<dataset_name>
```
where `<dataset_name>` is one of the datasets defined in `scripts/conf/dataset`. This will generate the corresponding Pytorch dataset, split it into training and test set according to the `test_size` defined in the dataset config file (default is 0.1, i.e. 10% used for testing), and save them to `output/datasets/<dataset_name>_train_val` and `output/datasets/<dataset_name>_test`, respectively.

## Model training and testing

To train and test a neural network-based forward model on your generated datasets, switch to the `scripts` directory and run
```
python train_eval.py dataset=<dataset_name> model=<model_name> transforms=<transform_name> task=train+eval task.final_eval=true
```
This will use the test set for final model evaluation. To instead use the validation set (small held-out subset of the training dataset) for evaluation, e.g. when tuning hyperparameters, use `task.final_eval=false`.

If you already have a trained model, stored in `<model_path>`, you can generate and evaluate predictions without retraining by running
```
python train_eval.py dataset=<dataset_name> model=<model_name> transforms=<transform_name> task=eval task.final_eval=true task.model_path=<model_path>
```

## Hyperparameter tuning

- PCA dimension
- Model architecture



