#!/bin/bash

DATA_DIR='./datasets/htr_lising_testing_data' 
LOG_DIR='./logs'
DETECTOR='DeDocDetector'
RECOG_MODEL='RuTrOCR'


CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/good_data/ --log_file $LOG_DIR/$DETECTOR_$RECOG_MODEL/log_good.txt \
    --detector $DETECTOR --recog_model $RECOG_MODEL

CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/middle_data/ --log_file $LOG_DIR/$DETECTOR_$RECOG_MODEL/log_middle.txt \
    --detector $DETECTOR --recog_model $RECOG_MODEL

CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/random_data/ --log_file $LOG_DIR/$DETECTOR_$RECOG_MODEL/log_random.txt \
    --detector $DETECTOR --recog_model $RECOG_MODEL
