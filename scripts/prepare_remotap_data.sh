#!/bin/bash

BASE_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_random_ensemble_1e5_only_CO2I"
DATA_DIR="${BASE_DIR}/synth_chunked"
PREFIX="syn_meas_s5_random_ensemble_1e5_only_CO2I"
# DATA_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_vary_AOT_fine_1e2/synth"
# PREFIX="syn_meas_s5_vary_AOT_fine_1e2"
# DATA_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_extended_ensemble_1e5/synth_chunked"
# PREFIX="syn_meas_s5_extended_ensemble_1e5"
INSTRUMENT="instr1_noise0" # instr2_noise0 if MAP is also simulated
PROXY="ns" # include non-scattering spectra as proxy

NCHUNKS=$1


# for each band, run prepropressing
for BAND in "NIR" "SWIR1" "SWIR3"; do
    echo "Preprocessing ${BAND}..."
    TARGET_DIR="${BASE_DIR}/${BAND}_preprocessed"
    python prepare_remotap_data.py \
        --data_dir $DATA_DIR --target_dir $TARGET_DIR --prefix $PREFIX --band $BAND \
        --n_chunks $NCHUNKS --instrument $INSTRUMENT --postfix_proxy $PROXY
done