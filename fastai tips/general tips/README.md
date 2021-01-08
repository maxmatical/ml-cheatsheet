
# General Apporach to modelling

0. **(Optional) Try to find kaggle contests/other projects tackling the same problem with SOTA results**
1. start with the data you have, run a few models with some manual hyperparameter tuning to get a good starting point
2. if you need to improve performance, look to improve data first
  - collecting additional data
    - can leverage semi-supervised methods [here](https://github.com/maxmatical/fast.ai/tree/master/fastai%20tips/Semi-supervised%20learning)
  - cleaning/removing noisy data
  - fixing labels
 3. if no further improvements on data side (or cost/benefit too high), then run hyperparameter tuning
  - can try hyperparameter tuning on a subset of data for faster experimentation (only if subset gains => full data gains)

# fastai2 extension libraries
https://github.com/nestordemeure/fastai-extensions-repository

multi-gpu training in notebooks: https://github.com/philtrade/mpify

# production setting considerations
- have train/val/test set
- test set small set of gold standard data points

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

### **Ranger/General optimizer related tips:**
Ranger seems to work really well (try with both `fit_one_cycle` and `fit_fc`
  - `fit_one_cycle` may be better for transfer learning
![ranger](https://github.com/maxmatical/fast.ai/blob/master/ranger.png)

https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

https://github.com/mgrankin/over9000

For adam based optimizers in general, try setting eps =[1.0, 0.1, 0.01]. Change with
```
eps = 1e-4
learn.opt_func = partial(learn.opt_func, eps=eps)
```
[RangerAdabelief](https://forums.fast.ai/t/gradient-centralization-ranger-optimizer-updated-with-it/68420/18)

[RangerAdabelief repo](https://forums.fast.ai/t/gradient-centralization-ranger-optimizer-updated-with-it/68420/18)

[RangerAdabelief episilon values](https://github.com/juntang-zhuang/Adabelief-Optimizer#2-epsilon)

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
### Save best model vs reducing number of epochs
- might be better to re-train at reduced number of epochs (the epoch with best metric) instead of using `SaveModelCallback`
- want a model with low learning rate


### Early stopping
https://docs.fast.ai/callbacks.tracker.html#EarlyStoppingCallback

### **Handling imbalanced data**

- They found oversampling the rare class until it's equally frequent was the best approach in every dataset they tested
- paper: https://arxiv.org/abs/1710.05381
- fastai callback: https://forums.fast.ai/t/pytorch-1-3-breaks-oversamplingcallback/56488
```
from torch.utils.data.sampler import WeightedRandomSampler    
# callback
class OverSamplingCallback(LearnerCallback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.labels = self.learn.data.train_dl.dataset.y
        _, counts = np.unique(self.labels,return_counts=True)
        self.weights = torch.DoubleTensor((1/counts)[self.labels])
        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i] for i in range(len(self.learn.data.train_dl.dataset))])
        self.total_len_oversample = int(self.learn.data.c*np.max(self.label_counts))

    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(self.weights,self.total_len_oversample), self.learn.data.train_dl.batch_size,False)
        
learn.fit_one_cycle(args.n_epochs, 
                  args.lr, 
                  pct_start = args.pct_start,
                  callbacks=[OverSamplingCallback(learn),
                             SaveModelCallback(learn, every='improvement', monitor='f_beta', 
                                               name=f'{args.model}_classifier_stage1{use_mixup}{suffix}')])
```

**Note**: can try applying a threshold such that you don't get a 50/50 split (i.e. 70/30 class split may work better) by changing `self.total_len_oversample`

**Weighted loss function**
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
**focal loss (only for binary??)
```
class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()

```

**focal loss for 1 output:** 
```
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```

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
 
### Diagnose model performance
1. train model
2. predict on training data
3. Sort by loss/class confusion (eg diff between top k logits/predicted probs)
4. Use this as a guide to prioritize data collection
  - Relabel as needed

**Tip:** Can use class loss/confusion and generate unlabelled data pseudolabels (if you have unlabelled data) to prioritize classes for manual labelling
  - eg `cats` and `dogs` get confused a lot in validation data, predict on unlabelled dataset, and focus on predictions of `cats` and `dogs` for manual labelling