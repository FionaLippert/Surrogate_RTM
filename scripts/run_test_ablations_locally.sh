#!/bin/bash

source activate sron_pytorch

export TORCH_USE_CUDA_DSA=1

bands=(nir swir1 swir3)
latent_dims=(100 160 140)
n_train=(464 1000 2150 4640 10000 21500 46400 100000 215000 464000 1000000)
train_frac=(0.000464 0.001 0.00215 0.00464 0.01 0.0215 0.0464 0.1 0.215 0.464 1.0)

data_path="/nfs/Sentinel5/users/fional/gitlab/surrogate_rtm/notebooks/paper_data/ablations/training_data_size"


for i in "${!bands[@]}"; do
  band=${bands[$i]}
  dim=${latent_dims[$i]}

  for j in "${!n_train[@]}"; do
    n=${n_train[$j]}
    frac=${train_frac[$j]}

    #########################################################################

    echo "Testing/predicting $band radiances using base emulator with n_train=$n_train"

    model_path="${data_path}/${band}/mlp-pca/${n_train}/model.ckpt"

    python train_eval.py \
      device=local \
      model=MLP_PCA_tempPCA_constant_latent_dim \
      model.output_latent_dim=${dim} \
      dataset=synthetic_remotap_${band}_spectrometer_only_LHS_matched_retrieval_era5_paper \
      dataset.normalization.per_wavelength=true \
      logger.project=spectrometer_only_${band}_era5_paper_train_frac_eval \
      task=eval \
      task.data_frac=$frac \
      task.final_eval=true \
      task.save_predictions=true \
      task.model_path=$model_path \
      seed=0 \

    #########################################################################

    echo "Testing/predicting $band radiances using hybrid MLP-PCA emulator"

    model_path="${data_path}/${band}/ratio-mlp-pca/${n_train}/model.ckpt"

    python train_eval.py \
      device=local \
      model=MLP_PCA_Ratio_tempPCA_constant_latent_dim \
      model.output_latent_dim=${dim} \
      dataset=synthetic_remotap_${band}_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper \
      dataset.normalization.per_wavelength=true \
      logger.project=spectrometer_only_${band}_era5_paper_train_frac_eval \
      task=eval \
      task.data_frac=$frac \
      task.final_eval=true \
      task.save_predictions=true \
      task.model_path=$model_path \
      seed=0 \

    #########################################################################

    echo "Testing/predicting $band radiances using hybrid CNN emulator"

    model_path="${data_path}/${band}/ratio-cnn/${n_train}/model.ckpt"

    python train_eval.py \
      device=local \
      model=SequenceNNSurrogateRTMRatio_proxy_input_tempPCA_input_processing \
      dataset=synthetic_remotap_${band}_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper \
      dataset.normalization.per_wavelength=false \
      logger.project=spectrometer_only_${band}_era5_paper_train_frac_eval \
      task=eval \
      task.data_frac=$frac \
      task.final_eval=true \
      task.save_predictions=true \
      task.model_path=$model_path \
      seed=0 \

    done

done

echo "Done!"