model_name=maxvit_base_512_25d_all_train

python -m run.train \
  dataset.num_folds=5 \
  dataset.test_fold=0 \
  dataset.phase=valid \
  training.batch_size=16 \
  training.batch_size_test=16 \
  training.epoch=25 \
  augmentation.use_light_aug=true \
  preprocessing.h_resize_to=512 \
  preprocessing.w_resize_to=512\
  model.base_model=maxvit_base_tf_512.in21k_ft_in1k \
  training.num_workers=112 \
  training.num_gpus=4 \
  optimizer.lr=5e-4 \
  scheduler.warmup_steps_ratio=0.1 \
  training.use_wandb=false \
  model.use_label_aux_min_max=true \
  dataset.normalize_method="mean_std" \
  training.sync_batchnorm=false \
  training.use_amp=false \
  optimizer.type=adamw \
  dataset.n_frames_before=2 \
  model.use_25d=true \
  training.use_gradient_checkpointing=true \
  out_dir=../results/${model_name}

mkdir ../inference_models/${model_name}
cp $(find ../results/${model_name}/weights -name model_weights.pth) ../inference_models/${model_name}/model_weights.pth
cp ../results/${model_name}/.hydra/config.yaml ../inference_models/${model_name}/
