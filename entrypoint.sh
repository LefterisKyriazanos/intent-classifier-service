#!/bin/bash

# Define user arguments (if not provided)
CLASSIFIER=${CLASSIFIER:-'GPT'}
MODEL=${MODEL:-'gpt-3.5-turbo'}
CLASSIFIER_TYPE=${CLASSIFIER_TYPE:-'zero-shot'}
TRAIN_DS_PATH=${TRAIN_DS_PATH:-'./data/atis/train.tsv'}
TEST_DS_PATH=${TEST_DS_PATH:-'./data/atis/test.tsv'}
PORT=${PORT:-8080}

# Run the main Python script providing user arguments
python server.py --classifier "$CLASSIFIER" \
                 --model "$MODEL" \
                 --classifier_type "$CLASSIFIER_TYPE" \
                 --train_ds_path "$TRAIN_DS_PATH" \
                 --test_ds_path "$TEST_DS_PATH" \
                 --port "$PORT"