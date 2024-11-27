#!/bin/bash

DATA_DIR="/nfs/Sentinel5/users/fional/S5_test_data/s5_caseE_ensemble/"
PREFIX="ensemble"

# for each band, run prepropressing
for BAND in "BAND1" "BAND2" "BAND3"; do
    echo "Preprocessing ${BAND}..."
    TARGET_DIR="/nfs/Sentinel5/users/fional/S5_test_data/s5_caseE_ensemble/${BAND}_preprocessed/"
    python prepare_remotec_data.py --data_dir $DATA_DIR --target_dir $TARGET_DIR --prefix $PREFIX --band $BAND
done