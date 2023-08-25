model_name_pretrain=resnest269e_512_25d_pl_pretrain_all_train
model_name=resnest269e_512_25d_pl_pretrain_finetune_all_train

python -m run.train \
  dataset.num_folds=5 \
  dataset.test_fold=0 \
  dataset.phase=valid \
  training.batch_size=32 \
  training.batch_size_test=32 \
  training.epoch=50 \
  augmentation.use_light_aug2=true \
  preprocessing.h_resize_to=512 \
  preprocessing.w_resize_to=512 \
  model.base_model=resnest269e \
  training.num_workers=112 \
  training.num_gpus=4 \
  optimizer.lr=1e-4 \
  scheduler.warmup_steps_ratio=0.1 \
  training.use_wandb=false \
  model.use_label_aux_min_max=true \
  dataset.normalize_method="mean_std" \
  training.sync_batchnorm=false \
  dataset.n_frames_before=2 \
  model.use_25d=true \
  optimizer.type=adamw \
  dataset.augmentation.p_frame=1 \
  out_dir=../results/${model_name_pretrain}


python -m run.train \
  dataset.num_folds=5 \
  dataset.test_fold=0 \
  dataset.phase=valid \
  training.batch_size=32 \
  training.batch_size_test=32 \
  training.epoch=10 \
  augmentation.use_light_aug=true \
  preprocessing.h_resize_to=512 \
  preprocessing.w_resize_to=512 \
  model.base_model=resnest269e \
  training.num_workers=112 \
  training.num_gpus=4 \
  optimizer.lr=1e-5 \
  scheduler.warmup_steps_ratio=0.1 \
  training.use_wandb=false \
  model.use_label_aux_min_max=true \
  dataset.normalize_method="mean_std" \
  training.sync_batchnorm=false \
  dataset.n_frames_before=2 \
  model.use_25d=true \
  optimizer.type=adamw \
  dataset.augmentation.p_frame=0 \
  model.restore_path=../results/${model_name_pretrain}/weights/model_weights.pth \
  out_dir=../results/${model_name}


mkdir ../inference_models/${model_name}
cp $(find ../results/${model_name}/weights -name model_weights.pth) ../inference_models/${model_name}/model_weights.pth
cp ../results/${model_name}/.hydra/config.yaml ../inference_models/${model_name}/
