#!/bin/bash

# This script prepares the data for the surrogate RTM. It extracts the relevant
# data from the OPERA simulations and saves it in a format that can be used by
# the surrogate RTM dataloader.

# The script is called with three arguments:
# 1. BASE_DIR: the base directory where the OPERA simulations are stored
# 2. PREFIX: the prefix of the RemoTAP output files, e.g. "syn_meas_s5_random_ensemble_1e6"
# 3. NCHUNKS: the number of chunks to use (if simulated data is not split in chunks, set to 1)

BASE_DIR=$1
PREFIX=$2
NCHUNKS=$3

DATA_DIR="${BASE_DIR}/synth"

INSTRUMENT="instr1_noise2" # instr2_noise2 if MAP is also simulated
PROXY="ns" # include non-scattering spectra as proxy

script_path="/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/scripts"
python_path="/deos/fional/anaconda3/envs/sron_pytorch/bin/python"


# for each band, run prepropressing
for BAND in "NIR" "SWIR1" "SWIR3"; do
    echo "Preprocessing ${BAND}..."
    TARGET_DIR="${BASE_DIR}/${BAND}_preprocessed"
    $python_path $script_path/prepare_remotap_data.py \
        --data_dir $DATA_DIR --target_dir $TARGET_DIR --prefix $PREFIX --band $BAND \
        --n_chunks $NCHUNKS --instrument $INSTRUMENT --postfix_proxy $PROXY
done
