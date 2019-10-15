# fast.ai

collection of projects done using fastai as well as useful scripts and functions to improve results

# Things that can improve results

## General

### optimizer: radam

![radam in fastai](https://github.com/maxmatical/fast.ai/blob/master/radam.png)

For adam based optimizers in general, try setting eps =[1.0, 0.1, 0.01]

set bn_wd = False, true_wd = True

Average a bunch of models together trained with different seeds/hyperparameters

setting learner to fp16 Learner(data, model, metrics=[accuracy]).to_fp16()

Try using **SGD (momentum = 0.9, nesterov = True)**, can maybe generalize better

### **Ranger optimizer + extensions:**

https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

https://github.com/mgrankin/over9000

**Note:** run `learn.fit_fc()` with new optimizers (flat + cosine anneling)


### **Save best model**

```
learn.fit_one_cycle(10,
                   slice(lr/(2.6**4),lr), 
                   moms=(0.8,0.7),
                   callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                             name='best_classifier_final')])

```

### **Handling imbalanced data**

- They found oversampling the rare class until it's equally frequent was the best approach in every dataset they tested
- paper: https://arxiv.org/abs/1710.05381
- fastai callback: https://forums.fast.ai/t/pytorch-1-3-breaks-oversamplingcallback/56488

## NLP:

### **Label Smoothing**:

loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP


### Fixing mismatch between vocab size in data_clas and data_lm:

```
data_clas.vocab.itos = data_lm.vocab.itos

```
Before the following line
```
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
```
This has fixed the error.

### **Concatenating models to use metadata:**

https://towardsdatascience.com/next-best-action-prediction-with-text-and-metadata-building-an-agent-assistant-81117730be6b

and notebook https://www.kaggle.com/adai183/metadata-enhanced-text-classification

### **Can try using QRNN***

https://github.com/piegu/language-models 

https://github.com/piegu/language-models/blob/master/lm2-french.ipynb 

### **BERT + Fastai**

Medium article: https://medium.com/@abhikjha/fastai-integration-with-bert-a0a66b1cecbe 

Jupyter notebook: https://www.kaggle.com/abhikjha/jigsaw-toxicity-bert-with-fastai-and-fastai/notebook 

## CV:

### Misc
loss_func=LabelSmoothingCrossEntropy() for CV

use mixup learner = Learner(data, model, metrics=[accuracy]).mixup()

use test time augmentation


### **Res2net**

https://forums.fast.ai/t/res2net-with-some-improvements-and-implementation/54199

https://medium.com/@lessw/res2net-new-deep-learning-multi-scale-architecture-for-improved-object-detection-with-existing-de13095c9654


## Time Series

### **1D Resnet**:

https://github.com/tcapelle/TimeSeries_fastai 

https://forums.fast.ai/t/time-series-sequential-data-study-group/29686/331

https://github.com/timeseriesAI/timeseriesAI
