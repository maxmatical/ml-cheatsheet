## Semi-supervised learning/self training (for large unlabelled dataset and small labelled dataset)

### using pseudo-labels with fastai (expand training data)
https://isaac-flath.github.io/blog/deep%20learning/2020/12/15/Pseudo-Labeling.html

**Tip**: This same approach can be used on all (or some, based on level of confidence) of the unlabeled data to expand the training data. 

**Multiple ways of using unlabelled data**
1. (only if unlabelled data >> labelled data)
    - train on ALL pseudo-labelled data
    - fine tune on labelled data
    
2. train on labelled data + **ALL** of pseudo-labelled data

3. train on labelled data  + **SOME** of pseudo-labelled data, based on confidence (eg `predicted_proba >= 0.9`)

4. train on labelled data +  **ALL** pseudo-labelled data. Then further fine-tune on labelled data

 Not 100% sure which way is best
 
### removing noisy examples (low confidence) from pseudolabels may improve results
https://twitter.com/giffmana/status/1479015354366111746?t=pDWudblhtPkqgqxABdFlyQ&s=09&fbclid=IwAR2tHrrjDpDkEPfp1xFJWPux2vXgYkELUqJe8Llre7s8hjjRorg08mLGteA

- removing the lowest `k`% of pseudolabels by confidence (eg most confused examples) might improve results
- intuition: the most confused examples are noisy, low quality examples you don't want in training data
- works well in practice

### predicting soft pseudo-labels (distributions) to use for distillation/self training etc.
2 ways

1. array datasets with `n x k` matrix of preds as label

2. csv/df datasets with 
```
data | c1 | c2 | ... |cn
x    | y1 | y2 | ... |yn


data = (ImageList
    .from_df(path=path, df=df)
    .split_by_rand_pct()
    .label_from_df(cols=[c1,c2,..., cn],label_cls=FloatList)
    .transform(get_transforms(do_flip=False), size=(120,160))
    .databunch()
    .normalize(imagenet_stats))
    
### or in fastai v2



# defining batch and individual image transforms
size = 224
tfms = aug_transforms(size=size, max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1., p_lighting=1.)
item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.)) # used to get images all the same size

# batch transforms
batch_tfms = [*tfms, Normalize.from_stats(*imagenet_stats)]

# datablock
dblock_clas = DataBlock(blocks=(ImageBlock, RegressionBlock),
                   splitter=ColSplitter("is_valid"),
                   get_x=ColReader("data"),
                   get_y=ColReader([c1, c2, ..., cn]),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())
                   
dls = dblock_clas.dataloaders(df, bs=64)
# then use dls in learner


# for text
dblock_clas = DataBlock(blocks=(TextBlock.from_df('text', seq_len=72, vocab=dls.vocab), RegressionBlock),
                      get_x=ColReader("data"),
                      get_y=ColReader([c1, c2, ... , cn),
                      splitter=ColSplitter())
```


where `yi` is the predicted probablilty of class `ci`

and use `MSE` as loss function
 - maybe `nn.BCEWithLogitsLoss` or even `FocalLoss`, with `valid_loss` as monitor will work better
 - will `nn.CrossEntropyLoss` work?
 - `MSE` if output (`yi`) is logits, `BCE/CE` if output is predicted probability (b/w 0 and 1)

### custom accuracy function with soft psuedo-labels
use when psuedo-labels are soft probability distributions or logits
```
def custom_accuracy(pred, target):
    return (pred.argmax(dim=1) == target.argmax(dim=1)).float().mean()
```

### self supervised learning fastai extension
https://keremturgutlu.github.io/self_supervised/

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

### Example of leveraging self training with unlabeled data
https://twitter.com/ai_fast_track/status/1486187160042713091?t=xfB1lbLoLCUDJCSJMNWZXA&s=09&fbclid=IwAR10JhZs5k8NWxPhYJ49yNuIZZSQF9deEnyTryUJ2TTsvX_m2EsxTPPN5Mc
- interesting note: can manually validate/correct predictions in early stages of self training when model performance isn't as good

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

library for knowledge-distillation (can be used for self-distillation): https://nathanhubens.github.io/fasterai/

- note: will only work if all data is labelled, can't use when there's unlabelled data

### meta pseudo labels
[paper](https://arxiv.org/pdf/2003.10580.pdf)
MPL algorithm:
```
for t in range(epochs):
    1. train teacher network on labelled data (1 epoch)
    2. predict on unlabelled data
    3. train student network on unlabelled data (1 epoch)
    4. student predict on a validation set for labelled data to get y_pred_student
    5. use loss(y_val, y_pred_student) (nn.CrossEntropyLoss, etc.) as loss for teacher
    6. gradient update for teacher (gradient through gradient method)
    

```

### example of self-training/soft-pseudolabels
https://github.com/maxmatical/fast.ai/blob/master/fastai_hf_self_distillation.ipynb

- **Note**: according to [1](https://arxiv.org/abs/2010.16402) and [2](https://arxiv.org/abs/1906.02629), training with CE (no label smoothing) enables better feature representation for teachers. So the ideal training procedure would be
1. do `n` rounds of self-training (on labelled + unlabelled data) + finetuning on labelled data with `CrossEntropyLoss`
2. for final round of training, do self-training using `CrossEntropyLoss`, but fine-tune on only labelled data with label smoothing

- if doing self-distillation: do `n` rounds of self-distillation with `CrossEntropyLoss`, final round of self-distillation uses label smoothing
- if doing traditional distillation: train `n` teachers with `CrossEntropyLoss`. Only train student with label smoothing

### self-supervised learning with fastai
https://github.com/KeremTurgutlu/self_supervised

contains algos like DINO, BYOL, SwAV etc.

### Weakly supervised learning with Weasel
https://github.com/autonlab/weasel



## Self training for NLP (few-shot classification)

STraTA: Self-Training with Task Augmentation for Better Few-shot Learning
  - https://arxiv.org/abs/2109.06270
  - github: https://github.com/google-research/google-research/tree/master/STraTA
  - similar to other self training techniques

<img width="808" alt="image" src="https://user-images.githubusercontent.com/8890262/163197076-e64e19f1-a5e7-410e-bef0-e9cb45a145ba.png">

<img width="399" alt="image" src="https://user-images.githubusercontent.com/8890262/163197129-437cb2af-c632-4bed-9f6b-8f132489f58d.png">


## Synthetic Data Generation to Improve Model Performance
https://hackmd.io/gmDAH0fqRAKcZl3sPLdjsg

## COSINE: Fine-tuning pre-trained LMs without any labeled data
https://twitter.com/dvilasuero/status/1565580468862566400?t=hnpGI7QBKYbyYegdvWUA8w&s=09&fbclid=IwAR1Kb5aAHIaJzpSfQsTswi5U7IZlZBh6DCjJRt_T6jUfZth-NJELf6_bozw

## ASTRA: Self-training with Weak Supervision
https://twitter.com/dvilasuero/status/1565248371325026304?t=jnkGlrWhhbBJgjxkonEwag&s=09&fbclid=IwAR1Kb5aAHIaJzpSfQsTswi5U7IZlZBh6DCjJRt_T6jUfZth-NJELf6_bozw
