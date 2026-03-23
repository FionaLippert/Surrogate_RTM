#!/bin/bash
####################################################

###### command line options for cluster job submission ######
#$ -q Earth-cluster
#$ -q Earth-cluster@asterope.sron.nl   ### 50 cores
#$ -q Earth-cluster@celaeno.sron.nl   ### 50 cores
#$ -q Earth-cluster@merope.sron.nl   ### 50 cores
#$ -q Earth-cluster@electra.sron.nl   ### 28 cores


# run this script with:
# 
# qsub -l h_vmem=10G -pe Earth 32 -e /nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/scripts/qsub.err -o /nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/scripts/qsub.out -q Earth-cluster find_pca_dimension.sh

# or simply:
# bash find_pca_dimension.sh

####################################################

module_path="/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm"
script_path="${module_path}/scripts"
python_path="/deos/fional/anaconda3/envs/sron_pytorch/bin/python"

export PYTHONPATH=$PYTHONPATH:$module_path

cd $script_path

# iterate over nir, swir1 and swir3 bands
for band in nir swir1 swir3; do
    echo "Running PCA evaluation for band ${band}..."

    $python_path find_pca_dimension.py \
        task=pca_eval \
        dataset=synthetic_remotap_${band}_spectrometer_only_OPERA-S5 \
        > find_pca_dimension.log 2>&1
done
