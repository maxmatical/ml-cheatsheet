# fast.ai

collection of projects done using fastai as well as useful resources to improve results

# Step 1

**Try to find kaggle contests/other projects tackling the same problem with SOTA results**

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

According to https://www.reddit.com/r/MachineLearning/comments/dhws0l/r_on_the_adequacy_of_untuned_warmup_for_adaptive/, it seems like AdamW is still the best way to go


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

### **Learning rate**

- Try a smaller learning rate for the unfrozen model 
  - up to learn.freeze_to(-2) can keep lr in last layer
  - learn.freeze_to(-2) -> lr/2
  - learn.unfreeze() -> lr/10 in the last layer (also divide the slice lr in earlier layers by same values)
  
### Hyperparameter tuning: using optuna to tune hyperparameters

https://medium.com/@crcrpar/optuna-fastai-tabular-model-001-55777031e288 

Hyperband https://gist.github.com/PetrochukM/2c5fae9daf0529ed589018c6353c9f7b

Hyperband example: [full_twitter_sentiment_transformers.ipynb](https://github.com/maxmatical/fast.ai/blob/master/full_twitter_sentiment_transformers.ipynb) or [hyperband_test.ipynb](https://github.com/maxmatical/fast.ai/blob/master/hyperband_test.ipynb)

### Batch size:

it seems like 32/64 is the best starting point

### Batch size finder

https://medium.com/@danielhuynh_48554/implementing-a-batch-size-finder-in-fastai-how-to-get-a-4x-speedup-with-better-generalization-813d686f6bdf

Github: https://github.com/DanyWind/fastai_bs_finder

### Gradient Accumulation

https://docs.fast.ai/train.html#AccumulateScheduler

### Distributed training with multiple GPUs

https://docs.fast.ai/distributed.html

### Cosine LR schedule with restarts

https://docs.fast.ai/callbacks.general_sched.html#TrainingPhase-and-General-scheduler

### NaNs in fp16():

set eps to 1e-4 (or higher, 1e-2, 0.1, or 1.0), or [lower lr by about 10x or so](https://forums.fast.ai/t/a-code-first-introduction-to-natural-language-processing-2019/50203/27)
https://forums.fast.ai/t/mixed-precision-training/29601/21

### LabelSmoothing loss on everything!

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

### Transformers (BERT, RoBERTa, etc.) with fastai
RoBERTa: https://medium.com/analytics-vidhya/using-roberta-with-fastai-for-nlp-7ed3fed21f6c

Huggingface transformers: https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2?fbclid=IwAR2_MQh2XzsKEKqwckIShH8-wO5io6rD1wXb4nTn3-eABj8boE9jOYd3zsM 

**Tip:** sometimes training the entire model can have equal or better performance than freezing and gradually unfreezing model

### **AWD-LSTM Specific**

- Higher drop_mult on LM learner (1.), then smaller dropout on classifier

- Ensemble a fwd and backwards bwd = True models
  - backwards model: https://github.com/fastai/course-nlp/blob/master/nn-vietnamese-bwd.ipynb 
  - ensemble model: https://github.com/fastai/course-nlp 

## CV:

### Misc
loss_func=LabelSmoothingCrossEntropy() for CV

use mixup learner = Learner(data, model, metrics=[accuracy]).mixup()

use test time augmentation


### **Res2net**

https://forums.fast.ai/t/res2net-with-some-improvements-and-implementation/54199

https://medium.com/@lessw/res2net-new-deep-learning-multi-scale-architecture-for-improved-object-detection-with-existing-de13095c9654

## Tabular data

### EDA

don't do too much EDA at the beginning, risk overfitting. Do "ml driven EDA"

### Working with dates

Use fastai [add_datepart](https://docs.fast.ai/tabular.transform.html#add_datepart) or [add_cyclic_datepart](https://docs.fast.ai/tabular.transform.html#add_cyclic_datepart)

### Saving dfs
save to feather `df.to_feather('...')`

### Speeding up experimentation/exploration

- use smaller subset of data
- use smaller rf (less trees)
- set_rf_sample() to let rf use subsets of data when fitting
- use full model/data at the last step

### Importance of a good validation set
- validation set needs to be representative of the test/deployment data
- can use test data to see how correlated is validation performance to test performance (only time to look at test data)

### Look at feature importance ASAP
- build a rf/gbm (doesn't have to be very accurate), then evaluate feature importance right after
- use either sklearn or SHAP
- try throwing away unimportant columns and refit a model -> should get similar (or slightly) better results, but much faster
  - re-run feature importance since colinearity is removed, makes feature importance a lot more clearer
  
### One hot encoding
- can be useful for low cardinality categorical variables (6-7 can be a good starting point)
- use `proc_df` function from old fastai structured
- may not improve performance, but can yield additional insight into feature importance

## Time Series

### **1D Resnet**:

https://github.com/tcapelle/TimeSeries_fastai 

https://forums.fast.ai/t/time-series-sequential-data-study-group/29686/331

https://github.com/timeseriesAI/timeseriesAI

## Audio

https://github.com/mogwai/fastai_audio

https://github.com/fastai/course-v3/blob/master/nbs/dl2/audio.ipynb

## Semi-supervised learning/self training (for large unlabelled dataset and small labelled dataset)

### Semi-weakly supervised learning

https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/ **also semi-supervised learning**

### Weakly and semi-supervised learning

https://ai.facebook.com/blog/mapping-the-world-to-help-aid-workers-with-weakly-semi-supervised-learning/

### Weakly supervised learning

https://engineering.fb.com/ml-applications/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/

### self training with noisy student

https://arxiv.org/abs/1911.04252 

### practical application of self training

https://arxiv.org/abs/1904.04445

### Don't use label smoothing for teachers (in distillation process)

https://medium.com/@lessw/label-smoothing-deep-learning-google-brain-explains-why-it-works-and-when-to-use-sota-tips-977733ef020 



