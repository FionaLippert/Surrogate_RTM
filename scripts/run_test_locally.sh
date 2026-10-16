#!/bin/bash

source activate sron_pytorch

export TORCH_USE_CUDA_DSA=1

# bands=(nir swir1 swir3)
# latent_dims=(100 160 140)

bands=(swir3)
latent_dims=(140)

for i in "${!bands[@]}"; do
  band=${bands[$i]}
  dim=${latent_dims[$i]}

  #########################################################################

  # echo "Testing/predicting $band radiances using base emulator"

  # model_path="/deos/fional/bitbucket/sron_notebooks/paper_models/${band}/mlp-pca/model.ckpt"

  # python train_eval.py \
  #   device=local \
  #   model=MLP_PCA_tempPCA_constant_latent_dim \
  #   model.output_latent_dim=${dim} \
  #   dataset=synthetic_remotap_${band}_spectrometer_only_LHS_matched_retrieval_era5_paper \
  #   dataset.normalization.per_wavelength=true \
  #   logger.project=spectrometer_only_${band}_era5_paper \
  #   task=eval \
  #   task.final_eval=true \
  #   task.save_predictions=true \
  #   task.model_path=$model_path \
  #   seed=0 \

  #########################################################################

  echo "Testing/predicting $band radiances using hybrid MLP-PCA emulator"

  model_path="/deos/fional/bitbucket/sron_notebooks/paper_models/${band}/ratio-mlp-pca/model.ckpt"

  python train_eval.py \
    device=local \
    model=MLP_PCA_Ratio_tempPCA_constant_latent_dim \
    model.output_latent_dim=${dim} \
    dataset=synthetic_remotap_${band}_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper \
    dataset.normalization.per_wavelength=true \
    logger.project=spectrometer_only_${band}_era5_paper \
    task=eval \
    task.final_eval=true \
    task.save_predictions=true \
    task.model_path=$model_path \
    seed=0 \

  #########################################################################

  echo "Testing/predicting $band radiances using hybrid CNN emulator"

  model_path="/deos/fional/bitbucket/sron_notebooks/paper_models/${band}/ratio-cnn/model.ckpt"

  python train_eval.py \
    device=local \
    model=SequenceNNSurrogateRTMRatio_proxy_input_tempPCA_input_processing \
    dataset=synthetic_remotap_${band}_ratio_spectrometer_only_LHS_matched_retrieval_era5_paper \
    dataset.normalization.per_wavelength=false \
    logger.project=spectrometer_only_${band}_era5_paper \
    task=eval \
    task.final_eval=true \
    task.save_predictions=true \
    task.model_path=$model_path \
    seed=0 \

done

echo "Done!"