#!/bin/bash
####################################################
# SRON Earth-cluster submission script
####################################################
#
##### To submit a job with 5GB memory and 16 cores, use:
# qsub -l h_vmem=5G -pe Earth 16 -e qsub.err -o qsub.out -q Earth-cluster qsub_job.sh
# This will write outputs to qsub.out and errors to qsub.err
#
#
##### options for cluster:
#$ -q Earth-cluster                     ### general queue
#$ -q Earth-cluster@asterope.sron.nl    ### 50 cores
#$ -q Earth-cluster@celaeno.sron.nl     ### 50 cores
#$ -q Earth-cluster@merope.sron.nl      ### 50 cores
#$ -q Earth-cluster@electra.sron.nl     ### 28 cores


# run this script with:
# 
# qsub -l h_vmem=10G -pe Earth 32 -e /nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/scripts/qsub.err -q Earth-cluster qsub_find_pca_dimension.sh

####################################################

module_path="/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm"
script_path="${module_path}/scripts"
python_path="/deos/fional/anaconda3/envs/sron_pytorch/bin/python"

export PYTHONPATH=$PYTHONPATH:$module_path

cd $script_path

$python_path find_pca_dimension.py > qsub_find_pca_dimension.log 2>&1

