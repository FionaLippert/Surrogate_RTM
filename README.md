# Surrogate Radiative Transfer Model (RTM)
A code base to develop and evaluate neural network-based surrogate RTMs. 
The aim is to use this to accelerate trace gas and aerosol retrievals.

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