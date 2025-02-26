# self-supervised Trajectory Representation Learning with multi-scale spatio-temporal feature exploration（ICDE-2025 Accepted）
The Implementation of self-supervised Trajectory Representation Learning with multi-scale spatio-temporal feature exploration （TrajRL）

![image](https://github.com/user-attachments/assets/49fe8906-e0bf-4e9d-9d8c-f3e6b2c420a3)


## Model
The model's code is in TrajRL/libcity/model/*. All json files are Configuration Files.

The files in TrajRL/libcity/data, TrajRL/libcity/executor, TrajRL/libcity/evaluator are used for data loading, training and evaluation respectively.
## Requirements
Our code is based on python3.9 and pytorch1.12.1 and you need install other dependencies in file requirements.txt.
```shell
pip install -r requirements.txt
```
You can get datasets according to the experimental settings section in the paper.
## Run
Please run the file run_model.py with arguments. 

For example, pretrain TrajRL:
```shell
python run_model.py --model TrajRLContrastiveLM --dataset porto --config porto --gpu_id 0 --split true --masking_ratio 0.2 --distribution geometric --out_data_argument1 trun --out_data_argument2 shift
```
