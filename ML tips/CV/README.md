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
guide: https://jarvislabs.ai/blogs/multiGPUs (timmm + multiple gpus)

https://walkwithfastai.com/vision.external.timm#Bringing-in-External-Models-into-the-Framework

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
- reaching top of lbs with some tricks (blurpool, hyperparam tuning, ranger, labelsmoothing, resnext, etc.)[https://radekosmulski.com/how-to-reach-the-top-of-the-imagenette-leaderboard/]

### View misclassifications
https://docs.fast.ai/widgets.class_confusion.html

### Object Detection: use IceVision library + fastai
https://airctic.com/getting_started/

https://airctic.com/retinanet/

### Timm and unets
https://www.youtube.com/watch?v=552FVdcHIUU

### BlurPool for shift invariance
[example notebook](https://github.com/maxmatical/ml-cheatsheet/blob/master/imagenette%20-%205%20epochs.ipynb)


### Dealing with imbalance in object detection
https://twitter.com/ai_fast_track/status/1488353683461095424?t=31ss-zCN4RTnoze9VyHi_A&s=09&fbclid=IwAR0UDEKPy_GmAPq5N-RBs0ASxhR8YrKQ5h_gZ1OqeCzu1_7gEIrhZgN2nCI

### Blazingly Fast Computer Vision Training with the Mosaic ResNet and Composer
https://www.mosaicml.com/blog/mosaic-resnet?src=jfrankle

### FFCV dataloader to speed up training
https://docs.ffcv.io/
- image data only

### Which image model checkpoints to use?
- kaggle notebook: https://www.kaggle.com/code/jhoward/which-image-models-are-best
- 

### fastxtend (extended tools for fastai)
https://github.com/warner-benjamin/fastxtend

### focalnet (object detection)
- https://twitter.com/jw2yang4ai/status/1591735758729404417?t=piNM1I3p8FgiL5LTp8LB-w&s=09&fbclid=IwAR2pBEy8OiPUas6u6vVEx7biNQOfju0ITnundDX0KXuVEm1zsmqQlqGEVHk
- SOTA on coco

### mosaic ml image segmentation recipes for 5x faster training
- https://www.mosaicml.com/blog/mosaic-image-segmentation
