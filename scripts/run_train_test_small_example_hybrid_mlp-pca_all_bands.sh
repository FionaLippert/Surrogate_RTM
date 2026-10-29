#!/bin/bash

source activate rtm_env

export TORCH_USE_CUDA_DSA=1
nvidia-smi

bands=(nir, swir1, swir3)
latent_dims=(100, 160, 140)

for i in "${!bands[@]}"; do
  band=${bands[$i]}
  dim=${latent_dims[$i]}

  echo "Training Hybrid MLP-PCA for $band"

    python train_eval.py device=local \
        dataset=synthetic_remotap_${band}_ratio_base_small_example \
        model=MLP_PCA_Ratio_tempPCA_constant_latent_dim \
        model.output_latent_dim=${dim} \
        logger.project=spectrometer_only_${band}_small_example \
        transforms=log_targets_log_proxy_log_albedo_era5 \
        task.final_eval=true \
        task.save_predictions=true \
        seed=0 \
        task.data_frac=1.0 \
        trainer.max_epochs=5 \
        model.lr=1e-3

done