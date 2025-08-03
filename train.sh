#!/bin/bash
source activate ZTOF
# python -u main.py --dataset_name oxford_pet --num_workers 8
# python -u main_cub.py --dataset_name cub --num_workers 8
# python -u main_pet.py --dataset_name oxford_pet --num_workers 8
python -u main_food.py --dataset_name food101 --num_workers 8
#bash train.sh > pet.log 2>&1
#bash train.sh > cub.log 2>&1
#bash train.sh > food.log 2>&1
