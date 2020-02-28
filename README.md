# fast.ai

collection of projects done using fastai as well as useful resources to improve results

# Step 1

**Try to find kaggle contests/other projects tackling the same problem with SOTA results**

# Things that can improve results

## General

### FP16
setting learner to fp16 Learner(data, model, metrics=[accuracy]).to_fp16()


### Model ensembling

Average a bunch of models together trained with different seeds/hyperparameters

### **Ranger/General optimizer related tips:**
Ranger seems to work really well
![ranger](https://github.com/maxmatical/fast.ai/blob/master/ranger.png)

https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

https://github.com/mgrankin/over9000

For adam based optimizers in general, try setting eps =[1.0, 0.1, 0.01]. Change with
```
eps = 1e-4
learn.opt_func = partial(learn.opt_func, eps=eps)
```

**set bn_wd = False, true_wd = True**

**Note:** run `learn.fit_fc()` with new optimizers (flat + cosine anneling)

According to https://www.reddit.com/r/MachineLearning/comments/dhws0l/r_on_the_adequacy_of_untuned_warmup_for_adaptive/, it seems like AdamW is still the best way to go

Try using **SGD (momentum = 0.9, nesterov = True)**, can maybe generalize better


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
  
 - [Choose 10x less LR than lowest point in LR Finder](https://forums.fast.ai/t/how-to-choose-the-learning-rate/61931/5)
  
### Hyperparameter tuning: using optuna/hyperband to tune hyperparameters

https://medium.com/@crcrpar/optuna-fastai-tabular-model-001-55777031e288 

Hyperband https://gist.github.com/PetrochukM/2c5fae9daf0529ed589018c6353c9f7b

Hyperband/Optuna example: [full_twitter_sentiment_transformers.ipynb](https://github.com/maxmatical/fast.ai/blob/master/full_twitter_sentiment_transformers.ipynb) or [hyperband_test.ipynb](https://github.com/maxmatical/fast.ai/blob/master/hyperband_test.ipynb)

[Optuna + hyperband](https://optuna.readthedocs.io/en/latest/reference/pruners.html)

[Optuna + fastai example](https://github.com/optuna/optuna/blob/master/examples/fastai_simple.py). Can change ` optuna.pruners.MedianPruner()` to `optuna.pruners.HyperbandPruner()`

Load model with best hyperparameters

```
model.set_params(**study.best_params)
model.fit(X, y)
```

### Batch size:

it seems like 32/64 is the best starting point

### Batch size finder

https://medium.com/@danielhuynh_48554/implementing-a-batch-size-finder-in-fastai-how-to-get-a-4x-speedup-with-better-generalization-813d686f6bdf

Github: https://github.com/DanyWind/fastai_bs_finder

### Gradient Accumulation

https://docs.fast.ai/train.html#AccumulateScheduler

[Gradient accumulation for mixed precision](https://forums.fast.ai/t/gradient-accumulation-with-fp16-training/61263/2)

[Example Notebook of grad accumulation](https://github.com/maxmatical/fast.ai/blob/master/Gradient_Accumulation_tests.ipynb)

### Distributed training with multiple GPUs

https://docs.fast.ai/distributed.html

### Cosine LR schedule with restarts

https://docs.fast.ai/callbacks.general_sched.html#TrainingPhase-and-General-scheduler

### NaNs in fp16():

set eps to 1e-4 (or higher, 1e-2, 0.1, or 1.0), or [lower lr by about 10x or so](https://forums.fast.ai/t/a-code-first-introduction-to-natural-language-processing-2019/50203/27)
https://forums.fast.ai/t/mixed-precision-training/29601/21

### LabelSmoothing loss on everything!

### Proper size of validation set

Run model with same hyperparameters 5 times, see the `std` of metric as well as standard error (`std/sqrt(n)`) to see how consistent it is  

### Useful pytorch libraries
[pytorch catalyst](https://github.com/catalyst-team/catalyst)

### Productionizing models
[jit + quantization](https://forums.fast.ai/t/using-torch-quantization/56582)
 - use try static quantization `torch.quantization.quantize`

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

### General
loss_func=LabelSmoothingCrossEntropy() for CV

use mixup learner = Learner(data, model, metrics=[accuracy]).mixup()

use test time augmentation

### More CNN archs (Resnext, SENet etc.)
https://github.com/PPPW/deep-learning-random-explore/blob/master/CNN_archs/cnn_archs.ipynb

### **Res2net**

https://forums.fast.ai/t/res2net-with-some-improvements-and-implementation/54199

https://medium.com/@lessw/res2net-new-deep-learning-multi-scale-architecture-for-improved-object-detection-with-existing-de13095c9654

### Image segmentation with attention
can try using `self_attention = True`

### Image captioning
https://github.com/fg91/Neural-Image-Caption-Generation-Tutorial

### Imagenette (and vairants) leaderboards 
https://github.com/fastai/imagenette 

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
- validation set needs to be representative of the test/deployment data (70/15/15 split is a good starting pt)
- can use test data to see how correlated is validation performance to test performance (only time to look at test data)
  1. build 5 models (varying in how good they perform on validation data)
  2. predict on both validation and test dataset
  3. plot val_score vs test_score, see how well is the correlation
- if there is a temporal aspect to the data, ALWAYS split by time
  - after getting a good model on validation data, retrain the same model (same hyperparameters) on train + val data (**for temporal data**)

### Look at feature importance ASAP
- build a rf/gbm (doesn't have to be very accurate), then evaluate feature importance right after
- use either sklearn or SHAP
- try throwing away unimportant columns and refit a model -> should get similar (or slightly) better results, but much faster
  - re-run feature importance since colinearity is removed, makes feature importance a lot more clearer
  
### One hot encoding
- can be useful for low cardinality categorical variables (6-7 can be a good starting point)
- use [`proc_df`](https://github.com/fastai/fastai/blob/master/old/fastai/structured.py) function from old fastai structured
- may not improve performance, but can yield additional insight into feature importance

### Removing redundant features
- use dendrograms (ON ONLY THE INTERESTING FEATURES IF YOU DID FEATURE IMPORTANCE BEFOREHAND)
- REMOVE REDUNDANT FEATURES AFTER FEATURE IMPORTANCE
```
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, # note df_keep.columns is after removing non-important features
      orientation='left', leaf_font_size=16)
plt.show()

```
- the further the splits are to the bottom (or right) of the plot means they are more closely related
- Drop columns one at a time and see if validation score improves or drops
  - if only drops a little bit, but makes model simpler, can go with that option (tradeoff a bit of performance for speed)
  - Don't drop all columns in a group, even if dropping each column individually doesn't affect performance much
  
### Partial dependence plots
- looking at relationship between feature and label when all other features are the same
- more useful for understanding data, rather than for predictive power
```
from pdpbox import pdp
from plotnine import *
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
def plot_pdp(feat, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, feat) # m here is the trained rf model, x is the x_train or sample of x_train
    return pdp.pdp_plot(p, feat_name, plot_lines=True, 
                        cluster=clusters is not None, 
                        n_cluster_centers=clusters)plot_pdp('YearMade')
                        
# then
plot_pdp('YearMade', clusters=5) 

# interaction plots between 2 features
feats = ['saleElapsed', 'YearMade']
p = pdp.pdp_interact(m, x, feats)
pdp.pdp_interact_plot(p, feats)
```

### Extrapolation for time dependent data (if test/live data is time dependent)

Only for tree based models (issues with extrapolation)

Remove time dependent variables from the model
1. Create label for `is_test = 1` or `is_test = 0`
2. Train model to predict `is_test`
3. Look at feature importance to see which features are most time sensitive
4. Remove each feature **one at a time** to see if improves performance on validation data
5. Remove the unhelpful features

Alternative: use NNs (can easily handle extrapolation into future) **OR** detrend data (with differencing)

### General Procedure
1. start with RF for steps above
2. Move on to GBT/NN after feature engineering

### Tabnet
[Discussion](https://forums.fast.ai/t/tabnet-with-fastai-v2/62600)

[Fastai v2 Implementation](https://github.com/mgrankin/fast_tabnet) 

### Shap + Fastai

https://github.com/muellerzr/fastai2-SHAP 

## Time Series classification

### **1D Resnet**:

https://github.com/tcapelle/TimeSeries_fastai 

https://forums.fast.ai/t/time-series-sequential-data-study-group/29686/331

https://github.com/timeseriesAI/timeseriesAI

## Time series forecasting
### N-BEATS
article https://towardsdatascience.com/n-beats-beating-statistical-models-with-neural-nets-28a4ba4a4de8
paper https://arxiv.org/abs/1905.10437

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
![noisy student diagram](https://github.com/maxmatical/fast.ai/blob/master/images/noisy-student.jpg)

### practical application of self training

https://arxiv.org/abs/1904.04445

### Don't use label smoothing for teachers (in distillation process)

https://medium.com/@lessw/label-smoothing-deep-learning-google-brain-explains-why-it-works-and-when-to-use-sota-tips-977733ef020 

### Fastai article on self-supervised learning
https://www.fast.ai/2020/01/13/self_supervised/

### Self-distillation
Similar to self-learning (but same model architecture) (i.e. teacher-student with same model)

https://arxiv.org/abs/2002.05715

![self-distillation](https://github.com/maxmatical/fast.ai/blob/master/images/self-distillation.jpg)

[Pytorch distllation example](https://github.com/peterliht/knowledge-distillation-pytorch)

[Mnist example with trained teacher](https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/mnist/distill_mnist.py)

### Self-supervised learning for CV imagenette using fastai2 (inpainting)
https://github.com/JoshVarty/SelfSupervisedLearning/blob/34ab526d39b31f976bc821a4c0924db613c2f7f5/01_InpaintingImageWang/03_ImageWang_Leadboard_192.ipynb

