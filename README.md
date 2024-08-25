# TrajRL
The Implementation of self-supervised Trajectory Representation Learning with multi-scale spatio-temporal feature
## Model
The model's code is in TrajRL/libcity/model/*. All json files are Configuration Files.
## Run
Please run the file run_model.py with arguments. 

Pre-training:
```shell
python run_model.py --model TrajRLContrastiveLM --dataset porto --config porto --gpu_id 0 --split true --masking_ratio 0.2 --distribution geometric --out_data_argument1 trun --out_data_argument2 shift
```
