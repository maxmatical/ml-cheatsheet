# fast.ai

collection of projects done using fastai as well as useful scripts and functions to improve results

## Things that can improve results
optimizer: radam
set bn_wd = False, true_wd = True
loss_func=LabelSmoothingCrossEntropy() for CV
loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP
setting learner to fp16 Learner(data, model, metrics=[accuracy]).to_fp16()
For CV: use mixup learner = Learner(data, model, metrics=[accuracy]).mixup()
