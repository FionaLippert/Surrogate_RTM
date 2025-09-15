#!/bin/bash
####################################################
# possible command line arguments:

###### give cluster list
#$ -q Earth-cluster
#$ -q Earth-cluster@asterope.sron.nl   ### 50 cores
#$ -q Earth-cluster@celaeno.sron.nl   ### 50 cores
#$ -q Earth-cluster@merope.sron.nl   ### 50 cores
#$ -q Earth-cluster@electra.sron.nl   ### 28 cores


# run this script with:
# qsub -l h_vmem=10G -pe Earth 32 -e /nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/scripts/qsub.err -q Earth-cluster qsub_compare_model_performance.sh


####################################################

module_path="/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm"
script_path="${module_path}/scripts"
python_path="/deos/fional/anaconda3/envs/sron_pytorch/bin/python"

export PYTHONPATH=$PYTHONPATH:$module_path

cd $script_path

$python_path compare_model_performance.py

