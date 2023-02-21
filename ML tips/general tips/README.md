
# General Apporach to modelling

0. **(Optional) Try to find kaggle contests/other projects tackling the same problem with SOTA results**
  - Especially important for novel tasks
  - If a well known task, can browse [papers with code sota](https://paperswithcode.com/sota) for sota papers
  - Also do a google search for `<TASK> state of the art` (eg `state of the art machine translation transformers`, `state of the art topic modelling sbert)
1. start with the data you have, run a few models with some manual hyperparameter tuning to get a good starting point
2. if you need to improve performance, look to improve data first
  - collecting additional data (**see section on data curation**)
    - can leverage semi-supervised methods [here](https://github.com/maxmatical/fast.ai/tree/master/fastai%20tips/Semi-supervised%20learning)
  - cleaning/removing noisy data
  - fixing labels
3. if no further improvements on data side (or cost/benefit too high), then run hyperparameter tuning
  - can try hyperparameter tuning on a subset of data for faster experimentation (only if subset gains => full data gains)

# Modelling tips from Kaggle
- https://twitter.com/lucasgvazquez/status/1550416693683699712?t=Q-bnVneXf6PygY-dDplT4w&s=09&fbclid=IwAR32pZBNOxUKAihnasi44tTgc_wyq73gw0WCLmt5qmSh22lB278DAd0YQ7Y
1. Establish good validation
2. Start with smaller models/data: goal is to iterate quickly on ideas
  - models: use smaller models like distilbert
  - data: use smaller images/text size (eg truncate `max_length` of tokenizer to `128` to start)
  - Here is where you try different ideas (synthetic data, data aug strategies, model strategies etc.)
3. When yo have exhausted all ideas, now it's the time to start scaling up
4. Getting the most out of your model:
  - Hyperparam optimization
  - Ensembling techniques: SWA, EMA, model souping, model averaging, stacking etc.


# Data curation
- An active process > just collecting data
- Want to collect more data for challenging/underperforming classes
- Collecting more data for high performing classes can hinder performance of other classes
  - If a class has lots of data (imbalanced relative to rest of data) and is performing well, can try removing some data to see if:
    1. performance of that class does not decrease
    2. performance of other classes improve
    If both cases are true, consider removing some data from that class
    
   
## Example data curation process
1. train on data
2. evaluation (per class p/r/f1 and overall accuracy) on validation data
  - look at underperforming classes
3. predict on unlabelled dataset (if there is one)
4. use unlabelled predictions to validate/improve training dataset
  - add incorrectly predicted examples back to training data with correct class
  - can add correct examples, but want to avoid collecting too many examples for high performing classes
  - **note:** can really only improve precision for that class, improving recall might require actively sourcing positive examples for that class

**note:** can also do step **3** on training data to monitor/improve training data quality

## Data curation distribution tips
- real-world data distribution is ~N(0,1)
- good dataset is ~U(-2,2)

Want to have balanced dataset across all classes, especially long tail

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

# Diagnosing DL model performance
https://twitter.com/rasbt/status/1565798671781961728
1. Make sure training loss converged
2. Check for overfitting
3. Compare accuracy to a zero-rule baseline
4. Look at failure cases
5. Plot at a confusion matrix

# production setting considerations
- have train/val/test set
- test set small set of gold standard data points

# Handling imbalanced data

good summary of dealing with imbalanced datsets: https://twitter.com/Fra_Pochetti/status/1518599651536027648?t=ODJKwqwBkdSVJLgjQslPuQ&s=09&fbclid=IwAR307j8N_57fTkkkL4C_5nnVBRmFBEJSPxnW0vMcMEHzbhmdyILAP2zIjpo

## naive over[sam](https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam)pling data
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
- based on this conversation: https://twitter.com/giffmana/status/1596176744763256834
  - may not perform better than standard BCE loss + pred prob calibration
- [for multi-class classification](https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss)
  - gamma is `gamma` and alpha is `weight` in constructor
  - set both = 1 for regular focal loss, and `alpha = 0.25, gamma = 2.` for weighted focal loss
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
  
```
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

### Cyclical focal loss
https://arxiv.org/abs/2202.08978

### Weighted Dataloader
samples data accorted to probability of appearing in batch
https://docs.fast.ai/callback.data.html#Datasets.weighted_dataloaders

WeightedDL example: https://www.kaggle.com/dienhoa/healthy-lung-classification-spectrogram-fast-ai

# Things that can improve results

## General
### Resource for fastai2 for various DL tasks
https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0


### FP16 training
- faster training (if using tensor cores)
- smaller model size (can use larger model)
- some regularization effects due to lower numerical precision

If dealing with instabilities in training
- if using an optimizer like `adam`, set `eps` to a higher value (eg `1e-8 -> 1e-7, 1e-6` etc.)
- skipping update steps where loss is `nan/inf`
- revert model checkpoints if training diverges
- last resort: train in fp32


### Model ensembling

Average a bunch of models together trained with different seeds/hyperparameters

easy way to snapshot ensemble:
- save models at end of training cycle as `stage1, stage2...`
- at inference, load multiple learners from each checkpoint and ensemble predictions

another way to ensemble: using stratified k-fold cv (to train k models), then ensemble models together. [see here](https://walkwithfastai.com/tab.cv)
- k-fold cv for imagewoof: https://walkwithfastai.com/Cross_Validation#What-is-K-Fold-Cross-Validation?

- ensembling via training same model on different losses

### Save best model and early stopping
- may not be the best idea
- better to adjust number of epochs or learning rate so that the best model is at the end of training

### Save best model/early stopping vs reducing number of epochs
- might be better to re-train at reduced number of epochs (the epoch with best metric) instead of using `SaveModelCallback`
- want a model with low learning rate at the end

### choosing LR

- somewhere between steepest point and min loss pt /10
```
lr_min, lr_steep = learn.lr_find()
print(f"choose lr between {lr_steep} and {lr_min/10}")
```

### Learning Rate schedulers
The following seem to be good choices
1. `FitOneCycle`
  - probably works well in most circumstances
2. `fit_flat_cos` 
  - Flat + cosine decay 
  - works well with ranger + variants
3. `ReduceLROnPlateau`
  - keeping patience fairly high (eg reduce lr as late as possible) might be beneficial
  - patience can be a hyperparameter
  - may work better when number of epochs is really high  
  - optional: add a linear/cosine decay at the end for last `x`% of training
  - optional: add SWA at the last `x`% of training
  - can also do `scheduler.step(train_loss)` every `k` steps of training (if data is really large and doing large number of epochs doesn't make sense)
    - or run validation step every `k` steps of training and `scheduler.step(val_loss/val_metric)`
4. [Flat LR + SWA](https://arxiv.org/abs/1803.05407?fbclid=IwAR0EctkySwLuvPJAlM-q31AIB9a6s8NSQPsh2ww6qJ7sbN1Z4TBJfHSwIP8)
  - Recommend to use flat LR + swa for last `x`% of training
  - **Could** work well with `ReduceLROnPlateau` because it's flat LR
    - **alternatively**: switch between LR scheduler and SWA. don't `schedule.step()` when swa is being used (can be a hyperparam to experiment with)
  - can be done either per epoch or every `k` steps after `x`%
```
# after k % of epochs, swa at the end of every epoch
swa_model = torch.optim.swa_utils.AveragedModel(model)
total_steps = n_epochs * len(train_dl)
scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)

for i in range(n_epochs):
    # check:
    # 1. k is > 0 (if 0, don't use swa at all)
    # 2. ith epoch is at k% pct of training
    use_swa: bool = (k > 0) and (i >= int(k * n_epochs))
    ##########
    # training step
    ##########
    for x, y in train_dls:
        pred = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()
        # only update lr schedule if not using SWA
        if not use_swa:
          scheduler.step()
    
    # only update parameters for swa if conditions are met
    if use_swa:
        swa_model.update_parameters(model)
        
    ##########
    # val step
    ##########
    # don't use swa yet
    for x, y in val_dls:
        preds = model(x)
        ...
            

# Update bn statistics for the swa_model at the end
torch.optim.swa_utils.update_bn(loader, swa_model)

# final validation check wtih SWA model now
if swa_model:
  model = swa_model
for x, y in val_dls:
  preds = model(x)
  # calc loss + metrics
  ....
print(swa_loss, swa_metric)
        
        
#########
alternative meethod
#########
# after x% of total training steps, swa update every k steps after
# using a different lr cycle

swa_model = torch.optim.swa_utils.AveragedModel(model)
total_steps = n_epochs * len(train_dl)
scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)

curr_step = 0
for i in range(n_epochs):
    ##########
    # training step
    ##########
    for x, y in train_dls:
        pred = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()
        
        # update curr_step
        curr_step += 1
    
        # only start swa after x% of total steps
        # and update every k steps
        if curr_step > int(x * total_steps) and curr_step % k == 0:
            swa_model.update_parameters(model)
            # skip updating lr scheduler
            if swa_disable_scheduler = True:
                continue
        scheduler.step()
            
    ##########
    # val step
    ##########
    # don't use swa in validation
    for x, y in val_dls:
        preds = model(x)
        ...
            
# outside of training loop
torch.optim.swa_utils.update_bn(loader, swa_model)


# note: make inferences with swa_model
# can also make ema models with

ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
        0.1 * averaged_model_parameter + 0.9 * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

will always be making val pred with ema_model in val step
```

### Efficiently Estimating Pareto Frontiers with Cyclic Learning Rate Schedules
https://twitter.com/MosaicML/status/1513561796833075207?t=ph2e6IGpZUBsKgu93R3i7g&s=09&fbclid=IwAR1MpsPY0-5bt0BRC2r3mggC7tFaWtVNP600ScrAATvoEJiBLMpRm7EVC0w

https://www.mosaicml.com/blog/efficiently-estimating-pareto-frontiers

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

- Tip for HPO: for hyperparameters that are percentages (eg dropout), use uniform distribution, otherwise (eg lr, wd, Adam eps) use log uniform scale

**note:** for pretrained models, save and reload weights every trial

Load model with best hyperparameters
```
model.set_params(**study.best_params)
model.fit(X, y)
```
### Potentially better hyperparameter optimization (hpo) with HEBO (NEURIPS 2020 optimization winner)
https://github.com/huawei-noah/HEBO 

### Alternative: LIPO vs Optuna
https://github.com/jdb78/lipo

### Google vizier for hpo
https://github.com/google/vizier

### Batch size:
- For speed: set bs to as large as will fit in GPU memory
- For generalization performance: it seems like 32/64 is the best for generalization (more updates, more noise for regularization)
  - this may not be particularly true, see https://www.reddit.com/r/MachineLearning/comments/un0crv/r_fullbatch_gd_generalizes_better_than_sgd/
  - possible that smaller bs performs well at `x` number of epochs, but training longer with larger bs might generalize just as well (which might be faster too)
- For contrastive learning (eg DPR, other in batch negatives), bs want to be as large as possible

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

**Note**: don't want to use label smoothing over CE when [pre-training](https://arxiv.org/abs/1906.02629) or when [training teachers in self-training/knowledge distillation](https://arxiv.org/abs/2010.16402)
- **ONLY** use label smoothing when fine-tuning. for pre-training, using `CrossEntropyLoss` acts as a better feature learner

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

pytorch loss func: https://erogol.com/online-hard-example-mining-pytorch/

idea: sort batch of `x, y` by `loss(x, y)`, and only take the top `k`% of the batch by loss (so model only sees the hard examples to use for backprop)

**ALT** it might actually be good to remove the top k% by loss. see https://twitter.com/giffmana/status/1479015354366111746?t=pDWudblhtPkqgqxABdFlyQ&s=09&fbclid=IwAR2tHrrjDpDkEPfp1xFJWPux2vXgYkELUqJe8Llre7s8hjjRorg08mLGteA
- idea: highest loss might actually be noisy, so remove the most noisy examples and learn with only high quality examples
- NOTE: loss != predicted probability, so maybe it will not work as well in practice

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

### fastai on tpu
https://github.com/butchland/fastai_xla_extensions

### 8 Bit optimizers
https://github.com/facebookresearch/bitsandbytes

saves up to 75% memory on training models

### Docker for DS/ML
https://muellerzr.github.io/fastblog/2021/12/05/DockerforDataScience.html

### Xformer library
https://devblog.pytorchlightning.ai/part-i-simplifying-transformer-research-with-xformers-lightning-a715737b8ad4

### Active learning with BaaL
https://devblog.pytorchlightning.ai/active-learning-made-simple-using-flash-and-baal-2216df6f872c

### Convert `nn.CrossEntropyLoss` to `nn.BCEWithLogitsLoss`
- could potentially work better as a 1-v-all problem
- use label smoothing could help as well (see pytorch BERT huggingface example)
- **NOTE** probably doesn't work on its own, but works better with mixup style data augmentation 

### Don't use early stopping/save best model callbacks!
- twitter thread: https://twitter.com/JFPuget/status/1558549407091625985?t=8o9iAodC8ES0MWu4Ws7x3A&s=09&fbclid=IwAR32tgirDraLvRmySmsQGL36IpTVMmdnlxIPvCmEpqKSCAESvPAg-GLvHh4
- applies both to DL, as well as some ML models (eg xgboost w/ early stopping)
- both can introduce too much variance to the model (**especially** when doing cross validation/ hpo)
- want model at the last epoch to be the best model
- 3 ways to make sure the best model is a the last epoch
  1. naive way: start with some number, if performance is still improving at the end, try increasing number of epochs. if performance starts to drop before the end (eg at epoch 35), set the number of epochs to the epoch with best performance
  2. add number of epochs as a hyperparameter, and run hpo with that
  3. alternatively, tune lr such that the best model is at the last epoch
 - overall goal: tune the last epoch result
  - number of epochs way require experimentation (i.e. no rules of thumb best practice for choosing  `n_epochs`)

### Exponential moving average (EMA) of model weights
https://github.com/fadel/pytorch_ema

see pytorch/pytorch lightning BERT notebook example

discussion on ema: https://www.reddit.com/r/MachineLearning/comments/ucflc2/d_understanding_the_use_of_ema_in_diffusion_models/
- notes:
- don't use ema model in val/train, only load in inference
- keep ema and regular model checkpoints
  - if at test time ema model doesn't perform well, but has good training/val numbers, try dropping ema model for regular model

### SWA + SAM
find flat-minima

paper: https://arxiv.org/pdf/2202.00661.pdf

twitter discussion: https://twitter.com/jeankaddour/status/1494437438856572932

efficient SAM: https://arxiv.org/abs/2203.02714

### Neat way to ensemble dropout in models
from this kaggle competition: https://www.kaggle.com/c/we-are-all-alike-on-the-inside/discussion/312371

sample code

```
class Model(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": True,
                "num_labels": self.num_labels,
            }
        )

        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, ids, mask, targets=None):
        transformer_out = self.transformer(ids, mask)
        pooled_output = transformer_out.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        if self.training:
          logits1 = self.output(self.dropout1(pooled_output))
          logits2 = self.output(self.dropout2(pooled_output))
          logits3 = self.output(self.dropout3(pooled_output))
          logits4 = self.output(self.dropout4(pooled_output))
          logits5 = self.output(self.dropout5(pooled_output))

          logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
          
        # during inference dropout isn't used so can bypass
        else:
          logits = self.output(pooled_output)
        return logits
        

```


### Maximal update parametrization (muP)
paper: https://arxiv.org/abs/2203.03466

repo: https://github.com/microsoft/mup

mutransformers: https://github.com/microsoft/mutransformers

muP in transformers repo?: https://github.com/huggingface/transformers/issues/16157

video: https://www.youtube.com/watch?v=iI_wsuXcj6Y

notes:
- most useful for expensive tasks (training from scratch, pre-training etc.)


### Averaging model weights
https://www.reddit.com/r/MachineLearning/comments/tcp8ya/r_model_soups_averaging_weights_of_multiple/

- like SWA but for different model runs, potentially better
- more computationally efficient than ensembling and competitive (may be slightly less performant in certain situations)
- works really well with fine-tuning
- greedy soup method: add models to the soup if adding it improves val metric 
- vary hyperparameters like: optimizer, data aug, lr, train iterations, random seed (or data shuffle)
- caution for averaging language models! many newer lms use weight tying for the output and embedding layers. averaging weights in this case can cause undesired behavior
- **key**: initialized weights need to be the same. if using pretrained weights -> can vary seed. if training from scratch -> keep seed same so initialized weights are the same
- **idea**: 
  1. run HPO and track all hyperparams + results. make sure initial weights is the same (careful about the linear layer, use seed_all or save the initialized weights)
  2. sort by metric (val acc etc.)
  3. starting from best model, for `n` models, re-train using hyperparams
  4. measure the new avg model (`new_souped_model = (k/(k+1) * best_souped_model + 1/(k+1) * cur_model`) vs `best_souped_model` where `k` is number of models currently in the soup
  5. if performance improves, make `best_soup_model = new_souped_model`
  - possible extension: hack optuna to save the top `n` models by metric, then use greedy souping

- **idea**: greedy souping within a single run?
  - similar to SWA, but instead of averaging last `k` models, average the **top** `k` models (via model checkpointing eg in pytorch lightning)
  - lr scheduler would also not be interrupted by SWA which keeps it constant

### MosaicML tips

https://www.mosaicml.com/blog/5-best-practices-for-efficient-model-training

Algorithms that can improve training performance: https://docs.mosaicml.com/en/latest/trainer/algorithms.html

### converting multiclass to multi-label
seen in resnet strikes back paper: https://arxiv.org/pdf/2110.00476.pdf

- works really well with mixup/cutmix
- use `torch.nn.BCEWithLogitsLoss` on logits (before sigmoid)
- may not actually be better than multiclass, but works because of mixup/cutmix data aug

### Polyloss instead of CrossEntropy or FocalLoss
- https://arxiv.org/abs/2204.12511
- twitter thread: https://twitter.com/tanmingxing/status/1519787578160869376
- pytorch impl: see [useful_loss_functions.py](https://github.com/maxmatical/ml-cheatsheet/blob/master/useful_loss_functions.py)
- note: doesn't seem to help much from my experiments

### Tracking gradients with `wandb.watch`
https://docs.wandb.ai/guides/integrations/pytorch#logging-gradients-with-wandb.watch

- tracking gradients may be useful to diagnose model performance
- more of an issue as model scales up? eg 1B+

### Curriculum learning inspired training ideas
- start with small batch size, grow to max batch size
  - similar idea (and can be combined with) progressive resizing (for text and images)
- can do multi phase eg bs=32, bs=64, bs=128 with 3 separate `trainer.fit` steps
  - this way because dataloader bs is static
- helps in 2 ways
  1. The loss function drops quite quickly in the very beginning, regardless of the number of examples to the model at each iteration. Since we decrease the number of computations in the beginning of training, we pass this stage of the loss plateau much faster.
  2. claims to stablize training
- note: growing batch size should only be useful if you're not training for multiple epochs (i.e. pre-training LLMs)? otherwise wall time per epoch may increase

### Learning with noisy labels
https://github.com/songhwanjun/Awesome-Noisy-Labels?fbclid=IwAR2y9hkYhvm2o8R5Bd_DNGkMnPyUTMm0hppvF6FbRV4PjzF6YEZZsoKOhEs

### Increasing numerical precision to remove spikes from loss curves
- https://twitter.com/_arohan_/status/1559327820546916353?t=y8-EHAzKX61s-XHrpspkaQ&s=09&fbclid=IwAR2BdqXlihxUAjOXsePRhGIS-zDxWwp3G1-__VWLADHkSsLH7h4A6FA-zMk
- if loss looks like this ![image](https://user-images.githubusercontent.com/8890262/184979560-fddba306-ff4c-4669-b504-a9fb85d34ed7.png)

- eg `float32 -> float64` before `log_softmax`
- NOTE: not sure if is solved by grad scaler (eg like in fp16) or would also work in that scenario


### using large batch sizes for contrastive loss when gpu memory is a bottleneck
1. use model paralellism (eg deepspeed zero3 + offload) to free up as much mem as possible for largest batch size
2. cache logits and targets per step, similar to gradient accumulation, until you get to a large enough batch size, then compute the loss with all the cache

### Various tools for python/pytorch etc.
https://github.com/stas00/toolbox?fbclid=IwAR3Kod863Qg4SYTfJl8qq9-s-VbRY0UpjG4K3JNCLlP4xmMeDOvrhc_arIE

### The WeightWatcher tool for predicting the accuracy of Deep Neural Networks
https://github.com/CalculatedContent/WeightWatcher


### LAWA (Latest weight averaging)
- https://twitter.com/jeankaddour/status/1578174175738404864
- speeds up training (not about performance like SWA)
- save last k checkpoints (after each epoch)

### using active learning
- https://twitter.com/__nmca__/status/1588575691284807682
- https://arxiv.org/abs/2211.01568

### initialize embedding layers (transformers) with normal distribution with std of 0.02 instead of 0.1
- https://twitter.com/borisdayma/status/1588636026482089984

### Compute-Efficient Deep Learning: Algorithmic Trends and Opportunities
- https://twitter.com/davisblalock/status/1601520350612770816

### Google research tuning playbook
https://github.com/google-research/tuning_playbook

### [Natural Language Processing with Disaster Tweets](https://chrwittm.github.io/posts/2023-01-17-nlp-with-disaster-tweets/)
- smaller batch sizes helps model train more quickly
- **train on ALL data (train + dev) after hyperparam tuning/optimization**

### Colossal-AI for large model training/inference
- github: https://github.com/hpcaitech/ColossalAI
- website: https://www.colossalai.org/
- pytorch lightning integration: https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html?highlight=colossal-ai#colossal-ai
  - https://www.hpc-ai.tech/blog/colossal-ai-0-2-0
  - more docs: https://github.com/hpcaitech/ColossalAI-Pytorch-lightning

### Using FSDP may be better if you can fit training into gpu mem
- i.e. no need for cpu offloading
- pros: more customizable, since using just pytorch code other than deepspeed/megatron/colossal ai etc.
- cons: no cpu offload

### Scaling model training w/ accelerate
https://huggingface.co/docs/transformers/v4.18.0/en/performance

### Activation checkpointing with HF models
- just use `model.gradient_checkpointing_enable()`

### Parameter efficient finetuning (PEFT) methods for LLMs
- https://github.com/huggingface/peft
- Lora, P-tuning (v2), prompt tuning
- p-tuning v2 may be comparable to full finetuning
- how does it stack up vs (mixture of) adapters?


### MosaicML streaming dataset
https://www.mosaicml.com/blog/mosaicml-streamingdataset

- fetch from S3
