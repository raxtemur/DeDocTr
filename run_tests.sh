#!/bin/bash

DATA_DIR='./datasets/htr_lising_testing_data'


CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/good_data/ --log_file log_good.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/middle_data/ --log_file log_middle.txt
CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/random_data/ --log_file log_random.txt