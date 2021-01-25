## Semi-supervised learning/self training (for large unlabelled dataset and small labelled dataset)

### using pseudo-labels with fastai (expand training data)
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
 - `MSE` if output is logits, `BCE/CE` if output is predicted probability (b/w 0 and 1)

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

### meta pseudo labels
[paper](https://arxiv.org/pdf/2003.10580.pdf)

