#!bin/bash

TRAIN=data/en-ud-train.conllu
DEV=data/en-ud-dev.conllu
TEST=data/en-ud-dev.conllu

LANGUAGE=en
LOSS=cross-entropy

bash complete_train_and_test_pipeline.sh $TRAIN $DEV $TEST $LANGUAGE $LOSS
