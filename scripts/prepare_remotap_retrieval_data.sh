#!/bin/bash

# This script prepares the data collected during retrievals for fine-tuning the surrogate RTM. 

# The script is called with two arguments:
# 1. DATA_DIR: the base directory where the RemoTAP retrieval data is collected
# 2. PREFIX: the prefix used to name the output files, e.g. "syn_meas_s5_random_ensemble_1e6"

DATA_DIR=$1
PREFIX=$2

script_path="/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/scripts"
python_path="/deos/fional/anaconda3/envs/sron_pytorch/bin/python"

NPIX=1000  # maximum number of pixels to process
MAX_CHI2=1000.0 # maximum chi2 value to consider a pixel/iteration for training


# for each band, run preprocessing
for BAND in "NIR" "SWIR1" "SWIR3"; do
    echo "Preprocessing ${BAND}..."
    TARGET_DIR="${DATA_DIR}/${BAND}_preprocessed"
    $python_path $script_path/prepare_remotap_retrieval_data.py \
        --data_dir $DATA_DIR --target_dir $TARGET_DIR --prefix $PREFIX \
        --band $BAND --npix $NPIX --max_chi2 $MAX_CHI2
done
