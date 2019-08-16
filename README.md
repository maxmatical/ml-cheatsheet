# fast.ai

collection of projects done using fastai as well as useful scripts and functions to improve results

## Things that can improve results

### General
optimizer: radam

![radam in fastai](https://github.com/maxmatical/fast.ai/blob/master/radam.png)

For adam based optimizers in general, try setting eps = 0.1 or eps = 0.01

set bn_wd = False, true_wd = True

Average a bunch of models together trained with different seeds/hyperparameters

setting learner to fp16 Learner(data, model, metrics=[accuracy]).to_fp16()

### NLP:
loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP

Fixing mismatch between vocab size in data_clas and data_lm:

```
data_clas.vocab.itos = data_lm.vocab.itos

```
Before the following line
```
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
```
This has fixed the error.


### CV:
loss_func=LabelSmoothingCrossEntropy() for CV

use mixup learner = Learner(data, model, metrics=[accuracy]).mixup()

use test time augmentation
