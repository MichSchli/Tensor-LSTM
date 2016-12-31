# Tensor-LSTM

This repository contains an implementation of the Tensor-LSTM dependency parser used in *arxiv link here*. The parser as described in the paper can be run on any UNIX system with:

```
bash complete_test_and_train_pipeline.sh $TRAIN $VALID $TEST $LANGUAGE $FORMAT
```

$TRAIN, $TEST, and $VALID represent the location of the train, test, and validation files. $LANGUAGE represents the string name of the language within the PolyGlot embedding format, and $FORMAT should take either the value "mse" or "cross-entropy" and represents the loss function.

We also provide a script and data for a single pilot test of the cross-entropy loss version of the parser using the English training and validation data in the Universal Dependencies treebank. This can be run as:

```
bash run_english_test.sh
```
