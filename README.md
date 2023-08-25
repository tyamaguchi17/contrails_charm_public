# 3rd Place Solution of Kaggle Contlails Competition
This is the charmq's part of the Preferred Contrail's solution.

[Competition page](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

[3rd place solution post](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430685)
## Prerequisite
```
pip install -r requirements.txt
```

## Directory
prepare empty directories: inference_models, results
```
$ ls ../
contrails_charm_public  data  inference_models  results
```

prepare dataset
```
$ ls ../data/
google-research-identify-contrails-reduce-global-warming  pseudo-label
```

prepare pseudo-labels (used in exp003)
```
$ ls ../data/pseudo-label
1000216489776414077  26936823957252040    4326677509020954479  5979520693683703833  7641797697439738821
1000603527582775543  2694055656055133422  4326863872205320288  598031925548118153   7642004243378384624
...
```

## Training
### exp000
2.5d maxvit base
```
$ bash exp/training/exp000_maxvit_base_2-5d.sh
```
The model file will be saved at `../inference_models/maxvit_base_512_25d_all_train/model_weights.pth`

### exp001
2.5d maxvit large
```
$ bash exp/training/exp001_maxvit_large_2-5d.sh
```
The model file will be saved at `../inference_models/maxvit_large_512_25d_all_train/model_weights.pth`

### exp002
2.5d maxvit base with heavier augmentation
```
$ bash exp/training/exp002_maxvit_base_2-5d_2.sh
```
The model file will be saved at `../inference_models/maxvit_base_512_25d_light_aug_2_all_train/model_weights.pth`

### exp003
2.5d resnest269e with pseudo labels. The training consists of two stages: pretraining and finetune.
```
$ bash exp/training/exp003_resnest269e_2-5d_pl.sh
```
The model file will be saved at `../inference_models/resnest269e_512_25d_pl_pretrain_finetune_all_train/model_weights.pth`


## Inference
The inference part should be executed after the training part.
### exp000
```
$ bash exp/inference/exp000_maxvit_base_2-5d.sh
```
The prediction for `validation` data will be saved at `../results/maxvit_base_512_25d_all_train_inference/test_results/test_results.npz`

### exp001
```
$ bash exp/inference/exp001_maxvit_large_2-5d.sh
```
The prediction for `validation` data will be saved at `../results/maxvit_large_512_25d_all_train_inference/test_results/test_results.npz`

### exp002
```
$ bash exp/inference/exp002_maxvit_base_2-5d_2.sh
```
The prediction for `validation` data will be saved at `../results/maxvit_base_512_25d_light_aug_2_all_train_inference/test_results/test_results.npz`

### exp003
```
$ bash exp/inference/exp003_resnest269e_2-5d_pl.sh
```
The prediction for `validation` data will be saved at `../results/resnest269e_512_25d_pl_pretrain_finetune_all_train_inference/test_results/test_results.npz`
