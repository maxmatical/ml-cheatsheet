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

### Object Detection: use IceVision library + fastai
https://airctic.com/getting_started/

https://airctic.com/retinanet/