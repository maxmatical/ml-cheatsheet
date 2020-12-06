# fast.ai

collection of projects done using fastai as well as useful resources to improve results



# Step 1

**Try to find kaggle contests/other projects tackling the same problem with SOTA results**

# fastai2 extension libraries
https://github.com/nestordemeure/fastai-extensions-repository

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
4. Relabel as needed

**Tip:** Can use class confusion on unlabelled data to prioritize classes for manual labelling

## NLP:

### Fastai2 with transformers:
https://github.com/morganmcg1/fasthugs

https://github.com/ohmeow/blurr

### data augmentation: back translate
https://amitness.com/2020/02/back-translation-in-google-sheets/

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
  
### Data augmentation for NLP
- translate text into another language, then translate back

### repo with transformer related code (training, inference, hpo, pseudolabel training)
https://github.com/maxmatical/fastai-transformers

## CV:

### General
loss_func=LabelSmoothingCrossEntropy() for CV

use mixup learner = Learner(data, model, metrics=[accuracy]).mixup()

use test time augmentation

### sample training script
![fastai_cnn_learner.py](https://github.com/maxmatical/fast.ai/blob/master/fastai_cnn_learner.py)

### transfer learning with ranger + fit_fc
Something like:
1. train head: `learn.freeze()` -> `lr`
2. train backbone: `learn.unfreeze()` -> between `lr/10` and `lr/50` (consider lowering pct_start to 0.1-0.3)
3. further training: `learn.unfreeze()` -> `lr/100`(consider lowering pct_start to 0.1-0.3)

### Progressive resizing
can speed up model training if needed

### More CNN archs (Resnext, SENet etc.)
https://github.com/PPPW/deep-learning-random-explore/blob/master/CNN_archs/cnn_archs.ipynb

### CNN archs with timm and fastai2 
guide: https://walkwithfastai.com/vision.external.timm#Bringing-in-External-Models-into-the-Framework

https://github.com/rwightman/pytorch-image-models

### **Res2net**

https://forums.fast.ai/t/res2net-with-some-improvements-and-implementation/54199

https://medium.com/@lessw/res2net-new-deep-learning-multi-scale-architecture-for-improved-object-detection-with-existing-de13095c9654

### Image segmentation with attention
can try using `self_attention = True`

### Image captioning
https://github.com/fg91/Neural-Image-Caption-Generation-Tutorial

### Imagenette (and vairants) leaderboards 
https://github.com/fastai/imagenette 

### View misclassifications
https://docs.fast.ai/widgets.class_confusion.html

### object detection 
https://airctic.com/getting_started/

https://airctic.com/retinanet/

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
cluster_columns(features) # new fastai2 feature
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

### NN Categorical embedding with high cardinality variable

- Don't want to have too many `cat_vars` with high cardinality, will take up lots of parameters in embedding layer (since each level needs its own embedding layer)
- Use RF to see if you can remove any of those vars without degreading performance

### General Procedure
1. start with RF for steps above
2. Move on to GBT/NN after feature engineering

### Tabnet
[Discussion](https://forums.fast.ai/t/tabnet-with-fastai-v2/62600)

[Fastai v2 Implementation](https://github.com/mgrankin/fast_tabnet) 

[Tabnet example](https://www.kaggle.com/syzymon/covid19-tabnet-fast-ai-baseline)

### Shap + Fastai

https://github.com/muellerzr/fastai2-SHAP 

### imputing missing features with MICE
[MICE](https://github.com/AnotherSamWilson/miceForest)

## Time Series classification

### Fastai extensions for timeseries:

https://github.com/timeseriesAI/tsai

https://github.com/tcapelle/TimeSeries_fastai 

https://forums.fast.ai/t/time-series-sequential-data-study-group/29686/331

https://github.com/ai-fast-track/timeseries

### Data agumentation for timeseries:
https://medium.com/@keur.plkar/from-sound-to-image-to-building-an-image-classifier-for-sound-with-fast-ai-3294909b3885

https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6

**Note**: may be very slow, for pure timeseires classification better to use this: https://github.com/timeseriesAI/tsai/blob/master/tutorial_nbs/03_Time_Series_Transforms.ipynb

## Time series forecasting
### N-BEATS
article https://towardsdatascience.com/n-beats-beating-statistical-models-with-neural-nets-28a4ba4a4de8
paper https://arxiv.org/abs/1905.10437

## Audio

https://github.com/mogwai/fastai_audio

https://github.com/fastai/course-v3/blob/master/nbs/dl2/audio.ipynb

## Semi-supervised learning/self training (for large unlabelled dataset and small labelled dataset)

### using pseudo-labels with fastai 
https://isaac-flath.github.io/blog/deep%20learning/2020/11/26/Pseudo-Labeling.html?fbclid=IwAR2WB6NAosbALCwB5HlIaNiCyRtQTjQEe5v1m8XjZAFyoLCppLC4-g5kj4I#What-Next?

**Tip**: This same approach can be used on unlabeled data to get data points the model is confident in to expand the training data. 

### predicting soft pseudo-labels (distributions) to use for distillation/self training etc.
2 ways

1. array datasets with `n x k` matrix of preds as label

2. csv/df datasets with 
```
data | c1 | c2 | ... |cn
x    | y1 | y2 | ... |yn


data = (ImageList
    .from_df(path=tub_path, df=df)
    .split_by_rand_pct()
    .label_from_df(cols=[c1,c2,..., cn],label_cls=FloatList)
    .transform(get_transforms(do_flip=False), size=(120,160))
    .databunch()
    .normalize(imagenet_stats))

```
where `yi` is the predicted probablilty of class `ci`

and use `MSE` as loss function
 - maybe `nn.BCEWithLogitsLoss` or even `FocalLoss`, with `valid_loss` as monitor will work better
 - will `nn.CrossEntropyLoss` work?

### Semi-weakly supervised learning

https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/ **also semi-supervised learning**

### Weakly and semi-supervised learning

https://ai.facebook.com/blog/mapping-the-world-to-help-aid-workers-with-weakly-semi-supervised-learning/


### Weakly supervised learning

https://engineering.fb.com/ml-applications/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/

### self training with noisy student

https://arxiv.org/abs/1911.04252 
![noisy student diagram](https://github.com/maxmatical/fast.ai/blob/master/images/noisy-student.jpg)
Note:
- Train student on both labelled and out of sample data -> create new dataset that combines both
- Can either use hard or soft pseudo labelling (soft pseudo labels might be slightly better)
- possible loss for soft speudo labels: https://discuss.pytorch.org/t/catrogircal-cross-entropy-with-soft-classes/50871


### practical application of self training

https://arxiv.org/abs/1904.04445

### Fastai article on self-supervised learning
https://www.fast.ai/2020/01/13/self_supervised/

### Fastai Rotation based self-supervised learning
https://amarsaini.github.io/Epoching-Blog/jupyter/2020/03/23/Self-Supervision-with-FastAI.html#FastAI-Vision-Learner-[Transfer-Classification]

### Self-distillation
Similar to self-learning (but same model architecture) (i.e. teacher-student with same model)

https://arxiv.org/abs/2002.05715

![self-distillation](https://github.com/maxmatical/fast.ai/blob/master/images/self-distillation.jpg)

**Note**: process is similar to psuedo-labels, just without any additional unlabelled data (predict soft-labels on itself to train)

### Self-supervised learning for CV imagenette using fastai2 (inpainting)
https://github.com/JoshVarty/SelfSupervisedLearning/blob/34ab526d39b31f976bc821a4c0924db613c2f7f5/01_InpaintingImageWang/03_ImageWang_Leadboard_192.ipynb

## Multi-task Learning
- [Example of multi-task learning with CNNs in fastai v1](https://gist.github.com/yang-zhang/ec071ae4775c2125595fd80f40efb0d6) 
- [Another example of multi-task learning on the same dataset (with NA's introduced to simulate missing data](https://nbviewer.jupyter.org/gist/denisvlr/802f980ff6b6296beaaea1a592724a51)


## Model deployment
### Speed up inference with jit and quantization
[jit + quantization](https://forums.fast.ai/t/using-torch-quantization/56582)
 - dynamic quantization: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dynamic_quantization_bert_tutorial.ipynb#scrollTo=IzyVSIKYIgN5
 - use try static quantization `torch.quantization.quantize`
 
quantize with
```
model = learn.model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
learn.model = quantized_model
```
### Using Dynamic and static quantization
https://spell.ml/blog/pytorch-quantization-X8e7wBAAACIAHPhT

### flask + gunicorn (easiest, not for scaling)
https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166
https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
`PYTHONPATH=. venv/bin/gunicorn -w 3 -t 600 --bind 192.168.0.215:4025 server:app`

### fastai + aws sagemaker 
https://github.com/fastai/course-v3/blob/master/docs/deployment_amzn_sagemaker.md

### fastai + torchserve + sagemaker
https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve

### fastai 1 + bentoml and kubernetes
https://course19.fast.ai/deployment_docker_kubernetes.html

### fastai2 + bentoml
https://docs.bentoml.org/en/latest/frameworks.html#fastai-v2

https://github.com/bentoml/gallery#fastai

### bentoml basics
https://docs.bentoml.org/en/latest/concepts.html?fbclid=IwAR3J05Bl7o5YLOF76v_WEIq1aAAgE0H0JJAphOr10VYuqf1qhfd0UKUIbs0
