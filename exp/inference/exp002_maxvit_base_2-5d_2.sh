model_name=maxvit_base_512_25d_light_aug_2_all_train

python -m run.train \
  dataset.num_folds=5 \
  dataset.test_fold=0 \
  dataset.phase=valid \
  training.batch_size=16 \
  training.batch_size_test=16 \
  training.epoch=25 \
  augmentation.use_light_aug2=true \
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
  test_model=../inference_models/${model_name}/model_weights.pth \
  out_dir=../results/${model_name}_inference
