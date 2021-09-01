
# General Apporach to modelling

0. **(Optional) Try to find kaggle contests/other projects tackling the same problem with SOTA results**
1. start with the data you have, run a few models with some manual hyperparameter tuning to get a good starting point
2. if you need to improve performance, look to improve data first
  - collecting additional data (**see section on data curation**)
    - can leverage semi-supervised methods [here](https://github.com/maxmatical/fast.ai/tree/master/fastai%20tips/Semi-supervised%20learning)
  - cleaning/removing noisy data
  - fixing labels
 3. if no further improvements on data side (or cost/benefit too high), then run hyperparameter tuning
  - can try hyperparameter tuning on a subset of data for faster experimentation (only if subset gains => full data gains)

# Data curation
- An active process > just collecting data
- Want to collect more data for challenging/underperforming classes
- Collecting more data for high performing classes can hinder performance of other classes
  - If a class has lots of data (imbalanced relative to rest of data) and is performing well, can try removing some data to see if:
    1. performance of that class does not decrease
    2. performance of other classes improve
    If both cases are true, consider removing some data from that class
    
## Adeptmind data curation process
1. train on data
2. evaluation (per class p/r/f1 and overall accuracy) on validation data
  - look at underperforming classes
3. predict on unlabelled dataset (if there is one)
4. use unlabelled predictions to validate/improve training dataset
  - add incorrectly predicted examples back to training data with correct class
  - can add correct examples, but want to avoid collecting too many examples for high performing classes
  - **note:** can really only improve precision for that class, improving recall might require actively sourcing positive examples for that class

**note:** can also do step **3** on training data to monitor/improve training data quality

# fastai2 extension libraries
https://github.com/nestordemeure/fastai-extensions-repository

multi-gpu training in notebooks: https://github.com/philtrade/mpify

 
# Diagnose model performance
1. train model
2. predict on training data
3. Sort by loss/class confusion (eg diff between top k logits/predicted probs)
4. Use this as a guide to prioritize data collection
  - Relabel as needed

**Tip:** Can use class loss/confusion and generate unlabelled data pseudolabels (if you have unlabelled data) to prioritize classes for manual labelling
  - eg `cats` and `dogs` get confused a lot in validation data, predict on unlabelled dataset, and focus on predictions of `cats` and `dogs` for manual labelling


# production setting considerations
- have train/val/test set
- test set small set of gold standard data points

# Handling imbalanced data

## naive oversampling data
- They found oversampling the rare class until it's equally frequent was the best approach in every dataset they tested
- paper: https://arxiv.org/abs/1710.05381


## Weighted loss function

### basic weighted loss for CE
`min_class` will have weight of 1
other classes will have weight of $sqrt(min_samples)/sqrt(n_samples)$
```
if weighted_loss:
    class_weights = []
    for c in learn.data.classes:
        class_weights.append(1/math.sqrt(len(df_train[df_train[LABEL_FIELD]==c])))
    max_weight = max(class_weights)
    class_weights = np.array(class_weights)/max_weight
    class_weights = torch.from_numpy(class_weights).float().cuda()
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    learn.loss_func = loss_func
print(f"using {learn.loss_func}")
```
### focal loss
- [fastai version (for multi-class classification)](https://docs.fast.ai/losses.html#FocalLossFlat)
  - gamma is `gamma` and alpha is `weight` in constructor
  - set both = 1 for regular focal loss, and `alpha = 0.25, gamma = 2.` for weighted focal loss
- weighted focal loss for **multi-label**
  ```
  class MultiLabelFocalLoss(nn.Module):
      def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
          """
          focal loss for multi label
          can set alpha, gamma = 1, 1 for non weighted
          reduction is either "mean", "sum", or None
          """
          super(FocalLoss, self).__init__()
          self.alpha = alpha
          self.gamma = gamma
          self.logits = logits
          self.reduction = reduction

      def forward(self, inputs, targets):
          if self.logits:
              BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
          else:
              BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
          pt = torch.exp(-BCE_loss)
          fl_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
          return fl_loss.mean() if self.reduce == 'mean' else fl_loss.sum() if self.reduce == 'sum' else fl_loss
  ```


### Weighted Dataloader
samples data accorted to probability of appearing in batch
https://docs.fast.ai/callback.data.html#Datasets.weighted_dataloaders

WeightedDL example: https://www.kaggle.com/dienhoa/healthy-lung-classification-spectrogram-fast-ai

# Things that can improve results

## General
### Resource for fastai2 for various DL tasks
https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0


### FP16

setting learner to fp16 Learner(data, model, metrics=[accuracy]).to_fp16()


### Model ensembling

Average a bunch of models together trained with different seeds/hyperparameters

easy way to snapshot ensemble:
- checkpoint learners at end of training cycle (`fit_fc` or `fit_one_cycle`) as `stage1, stage2...`
- at inference, load multiple learners from each checkpoint and ensemble predictions

another way to ensemble: using stratified k-fold cv (to train k models), then ensemble models together. [see here](https://walkwithfastai.com/tab.cv)
- k-fold cv for imagewoof: https://walkwithfastai.com/Cross_Validation#What-is-K-Fold-Cross-Validation?

### Optimizers
See `optimizers.py`

**Ranger** seems to work really well (try with both `fit_one_cycle` and `fit_flat_cos`)
  - `fit_flat_cos` seems to work better for Ranger
![ranger](https://github.com/maxmatical/fast.ai/blob/master/ranger.png)

https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

https://github.com/mgrankin/over9000

For adam based optimizers in general, try setting eps =[1.0, 0.1, 0.01]. Change with
```
eps = 1e-4
learn.opt_func = partial(learn.opt_func, eps=eps)
```
[**RangerAdabelief**](https://forums.fast.ai/t/gradient-centralization-ranger-optimizer-updated-with-it/68420/18)

[RangerAdabelief repo](https://github.com/juntang-zhuang/Adabelief-Optimizer)

[RangerAdabelief episilon values](https://github.com/juntang-zhuang/Adabelief-Optimizer#2-epsilon)

For **RangerAdabelief**, try `eps=1e-8` for CV, `eps=1e-16` for NLP, `eps=1e-12` for RL as default values (try a eps sweep for hyperparam tuning)

- **set `true_wd = True, bn_wd = False`**

- **Note:** run `learn.fit_flat_cos()` with new optimizers (flat + cosine anneling)

According to https://www.reddit.com/r/MachineLearning/comments/dhws0l/r_on_the_adequacy_of_untuned_warmup_for_adaptive/, it seems like AdamW may still be competitive

Try using **SGD (momentum = 0.9, nesterov = True) or RMSPROP(momentum=0.9)**, can maybe generalize better (try for CV, maybe also works for NLP)

[**Shapeness-Aware Minimization (SAM) optimizer**](https://github.com/davda54/sam) may be better for `ViT` and `MLP_Mixer` on cv tasks
  - use in fastai as a callback: https://github.com/maxmatical/ml-cheatsheet/blob/master/fastai_callbacks.py

Optimizers to try
```
optimizer_config_mapping = {
    "adamw": {"optimizer": AdamW, "wd": {"true_wd": True, "bn_wd": True}},
    "ranger": {"optimizer": Ranger, "wd": {"true_wd": True, "bn_wd": False}},
    "ranger_nogc": {
        "optimizer": Ranger,
        "hyperparameters": {"use_gc": False},
        "wd": {"true_wd": True, "bn_wd": False},
    },
    "rangeradabelief": {"optimizer": RangerAdaBelief, "wd": {"true_wd": True, "bn_wd": False}},
    "rangeradabelief_nogc": {
        "optimizer": RangerAdaBelief,
        "hyperparameters": {"use_gc": False},
        "wd": {"true_wd": True, "bn_wd": False},
    },
    "sgd": {
        "optimizer": SGD,
        "hyperparameters": {"momentum": 0, "nesterov": False},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
    "sgd_momentum": {
        "optimizer": SGD,
        "hyperparameters": {"momentum": 0.9, "nesterov": False},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
    "sgd_nesterov": {
        "optimizer": SGD,
        "hyperparameters": {"momentum": 0.9, "nesterov": True},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
    "rmsprop": {
        "optimizer": RMSprop,
        "hyperparameters": {"momentum": 0.9},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
}
```


### **Save best model**

```
learn.fit_one_cycle(10,
                   slice(lr/(2.6**4),lr), 
                   moms=(0.8,0.7),
                   callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', 
                                                             name='best_classifier_final')])

```
### Save best model vs reducing number of epochs
- might be better to re-train at reduced number of epochs (the epoch with best metric) instead of using `SaveModelCallback`
- want a model with low learning rate


### Early stopping
https://docs.fast.ai/callbacks.tracker.html#EarlyStoppingCallback


### choosing LR

- somewhere between steepest point and min loss pt /10
```
lr_min, lr_steep = learn.lr_find()
print(f"choose lr between {lr_steep} and {lr_min/10}")
```

### Learning rate tips for transfer learning

- decrease base learning rate for the unfrozen model 
  - up to `learn.freeze_to(-2)` can keep lr in last layer
  - `learn.freeze_to(-2)` and onwards -> lr/2
  - `learn.unfreeze()` -> base_lr between `[lr/10, lr/2]` in the last layer
  - divide the sliced lr in earlier layers by the same value
  
```
learn.freeze()
learn.fit_one_cycle(1, lr)
learn.unfreeze()
lr /=2
learn.fit_once_cycle(1, slice(lr/100, lr))

```
### Alternative LR strategy for transfer learning (more time consuming)
- instead of `lr /= 2`, run `learn.lr_find() at each stage of unfreezing
- might be better, but more time consuming

```
learn.freeze()
lr_min, lr_steep = learn.lr_find()
print(f"choose lr between {lr_steep} and {lr_min/10}")
lr = ...
learn.fit_one_cycle(1, lr)

learn.unfreeze()
lr_min, lr_steep = learn.lr_find()
print(f"choose lr between {lr_steep} and {lr_min/10}")
lr = ...
learn.fit_once_cycle(1, slice(lr/100, lr))


```

### Hyperparameter tuning: using optuna/hyperband to tune hyperparameters

https://medium.com/@crcrpar/optuna-fastai-tabular-model-001-55777031e288 

Hyperband https://gist.github.com/PetrochukM/2c5fae9daf0529ed589018c6353c9f7b

Hyperband/Optuna example: [full_twitter_sentiment_transformers.ipynb](https://github.com/maxmatical/fast.ai/blob/master/full_twitter_sentiment_transformers.ipynb) or [hyperband_test.ipynb](https://github.com/maxmatical/fast.ai/blob/master/hyperband_test.ipynb)

[Optuna + hyperband](https://optuna.readthedocs.io/en/latest/reference/pruners.html)

[Optuna + fastai example](https://github.com/optuna/optuna/blob/master/examples/fastai_simple.py). Can change ` optuna.pruners.MedianPruner()` to `optuna.pruners.HyperbandPruner()`

[ASHA](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner) might perform better than hyperband/PBT [source](https://arxiv.org/pdf/1810.05934.pdf) 

[Optuna + ray tune](https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7)

**note:** for pretrained models, save and reload weights every trial

Load model with best hyperparameters
```
model.set_params(**study.best_params)
model.fit(X, y)
```
### AutoML with microsoft NNI (supports population based training)
https://github.com/microsoft/nni

- [Population based training (PBT) with NNI](https://github.com/microsoft/nni/blob/master/docs/en_US/Tuner/PBTTuner.md)
- [Docs here](https://nni.readthedocs.io/en/latest/)
- supports NAS with pytorch 

### AutoML with autogluon
https://auto.gluon.ai/stable/index.html


### Batch size:
For speed: set bs to as large as will fit in GPU memory
For generalization performance: it seems like 32/64 is the best for generalization (more updates, more noise for regularization)

### Batch size finder

https://medium.com/@danielhuynh_48554/implementing-a-batch-size-finder-in-fastai-how-to-get-a-4x-speedup-with-better-generalization-813d686f6bdf

Github: https://github.com/DanyWind/fastai_bs_finder

### Gradient Accumulation

https://docs.fast.ai/train.html#AccumulateScheduler

[Gradient accumulation for mixed precision](https://forums.fast.ai/t/gradient-accumulation-with-fp16-training/61263/2)

[Example Notebook of grad accumulation](https://github.com/maxmatical/fast.ai/blob/master/Gradient_Accumulation_tests.ipynb)

### Distributed training with multiple GPUs

https://jarvislabs.ai/blogs/multiGPUs

### Cosine LR schedule with restarts

https://docs.fast.ai/callbacks.general_sched.html#TrainingPhase-and-General-scheduler

### NaNs in fp16():

set eps to 1e-4 (or higher, 1e-2, 0.1, or 1.0), or [lower lr by about 10x or so](https://forums.fast.ai/t/a-code-first-introduction-to-natural-language-processing-2019/50203/27)
https://forums.fast.ai/t/mixed-precision-training/29601/21

### LabelSmoothing loss on everything!

### Proper size of validation set

Run model with same hyperparameters 5 times, see the `std` of metric as well as standard error (`std/sqrt(n)`) to see how consistent it is  

### Speed up pytorch models
https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/

### multilabel stratified sampling

https://twitter.com/abhi1thakur/status/1357653979400982529?s=20

### Make sure `val` and `test` data performance is correlated
have data in `train/val/test`. what happens if `val` performance is high, but `test` performance is low?

1. check if `val` and `test` performance are correlated
  - eg if `val` goes up, does `test` go up? same with down
2. if no, need to reconstruct val data such that it's correlated with test data performance
3. if yes, just keep improving validation performance

### Prediction for long-tailed events
https://doordash.engineering/2021/04/28/improving-eta-prediction-accuracy-for-long-tail-events/

### random pytorch tips
https://www.reddit.com/r/MachineLearning/comments/n9fti7/d_a_few_helpful_pytorch_tips_examples_included/


1. Create tensors directly on the target device using the `device` parameter.
2. Use `Sequential` layers when possible for cleaner code.
3. Don't make lists of layers, they don't get registered by the `nn.Module` class correctly. Instead you should pass the list into a `Sequential` layer as an unpacked parameter.
4. PyTorch has some awesome objects and functions for distributions that I think are underused at `torch.distributions`.
5. When storing tensor metrics in between epochs, make sure to call `.detach()` on them to avoid a memory leak.
6. You can clear GPU cache with `torch.cuda.empty_cache()`, which is helpful if you want to delete and recreate a large model while using a notebook.
7. Don't forget to call `model.eval()` before you start testing! It's simple but I forget it all the time. This will make necessary changes to layer behavior that changes in between training and eval stages (e.g. stop dropout, batch norm averaging)

### hard example mining/batch loss filter
batch loss filter callback: https://github.com/maxmatical/ml-cheatsheet/blob/master/fastai_callbacks.py

### Dealing with positive-negative imbalance in multi-label data
- issue: if have a lot of classes, each class will see a lot of negative examples per 1 positive example
- solution: 
  - use a lower threhsold with BCE
  - use a different loss function than BCE
  - focal loss
  - [Asymmetric Loss For Multi-Label Classification](https://github.com/Alibaba-MIIL/ASL)

### training models that don't fit in GPU
- use DeepSpeed https://github.com/microsoft/DeepSpeed (model parallelism, ZeRO optimizer, etc.)
  - [getting started](https://www.deepspeed.ai/getting-started/)
- simple model parallelism https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

### using fastai in pytorch training loop
[example notebook](https://github.com/maxmatical/ml-cheatsheet/blob/master/imagenette_with_pytorch.ipynb)
- uses fastai loss functions and `flat_cos` learning rate scheduler 
