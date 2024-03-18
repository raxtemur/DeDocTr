#!/bin/bash

DATA_DIR='./datasets/htr_lising_testing_data' 
LOG_DIR='./logs'
DETECTOR='DeDocDetector'
RECOG_MODEL='AttnentionModel'
MODEL_NAME_OR_PATH='./source/attention_cyrillic_hkr_synthetic_stackmix/best_cer.pth'
CHARACTERS_FILE='./source/attention_cyrillic_hkr_synthetic_stackmix/character.txt'


CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/good_data/ --log_file $LOG_DIR/$DETECTOR"_"$RECOG_MODEL/log_good.txt \
    --detector $DETECTOR --recognition $RECOG_MODEL \
    --model_name_or_path $MODEL_NAME_OR_PATH --characters_file $CHARACTERS_FILE

CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/middle_data/ --log_file $LOG_DIR/$DETECTOR"_"$RECOG_MODEL/log_middle.txt \
    --detector $DETECTOR --recognition $RECOG_MODEL \
    --model_name_or_path $MODEL_NAME_OR_PATH --characters_file $CHARACTERS_FILE

CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_path $DATA_DIR/random_data/ --log_file $LOG_DIR/$DETECTOR"_"$RECOG_MODEL/log_random.txt \
    --detector $DETECTOR --recognition $RECOG_MODEL \
    --model_name_or_path $MODEL_NAME_OR_PATH --characters_file $CHARACTERS_FILE
