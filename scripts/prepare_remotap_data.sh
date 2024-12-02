#!/bin/bash

# DATA_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_random_ensemble_1e5/synth_chunked"
# PREFIX="syn_meas_s5_random_ensemble_1e5"
DATA_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_vary_AOT_fine_1e2/synth"
PREFIX="syn_meas_s5_vary_AOT_fine_1e2"
# DATA_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_extended_ensemble_1e5/synth_chunked"
# PREFIX="syn_meas_s5_extended_ensemble_1e5"
NCHUNKS=$1

# for each band, run prepropressing
for BAND in "NIR" "SWIR1" "SWIR3"; do
    echo "Preprocessing ${BAND}..."
    # TARGET_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_random_ensemble_1e5/${BAND}_preprocessed"
    TARGET_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_vary_AOT_fine_1e2/${BAND}_preprocessed"
    # TARGET_DIR="/deos/fional/remotap/simulations/RemoTAP_FWD/s5_extended_ensemble_1e5/${BAND}_preprocessed"
    python prepare_remotap_data.py --data_dir $DATA_DIR --target_dir $TARGET_DIR --prefix $PREFIX --band $BAND --n_chunks $NCHUNKS
done