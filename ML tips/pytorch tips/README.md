# pytorch specific tips
https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#5-consider-using-another-optimizer

## deepspeed
- used for model parallelism (when model doesn't fit on 1 gpu)
- contains zero optimizer
- use DeepSpeed https://github.com/microsoft/DeepSpeed (model parallelism, ZeRO optimizer, etc.)
  - [getting started](https://www.deepspeed.ai/getting-started/)
- simple model parallelism https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html


## fastai on tpu
https://github.com/butchland/fastai_xla_extensions

# integrating fastai functions

## using fastai in pytorch training loop
[example notebook](https://github.com/maxmatical/ml-cheatsheet/blob/master/imagenette_with_pytorch.ipynb)
- uses fastai loss functions and `flat_cos` learning rate scheduler 

## learning rate 
- lr finder: https://github.com/davidtvs/pytorch-lr-finder
- fit_flat_cos lr schedule (https://github.com/maxmatical/ml-cheatsheet/blob/master/imagenette_with_pytorch.ipynb)


## dataloaders
- use pytorch dataloaders `(train_dl, val_dl)`

## multi-gpu training (distributed data parallel)
- distributed training on multiple gpus: https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch

## mix precision training (amp)
- fp16 (mixed precision training): https://pytorch.org/docs/stable/notes/amp_examples.html
    - combining amp with lr schedulers: https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930
    - example of amp with `OneCycleLR` https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam

## Callbacks
- early stopping: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
- reduce lr on plateau callback: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    - test with using 2 lr schedulers
    - alternatively do something similar to early stopping, and use `optimizer.set_hyper('lr', new_lr)`
- terminate on nan callback

## mixup
- mixup: https://towardsdatascience.com/enhancing-neural-networks-with-mixup-in-pytorch-5129d261bc4a
  - call be a callback?

## SWA
- SWA: https://pytorch.org/docs/stable/optim.html

## SAM
- possibly more useful for CV than NLP
- add SAM wrapper for optimizer and modify training `if self.sam`: https://github.com/davda54/sam

## hard example mining
- batch loss filter/hard example mining callback
  - https://erogol.com/online-hard-example-mining-pytorch/

## loss function and metrics
- keep `FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1)` and `accuracy` from fastai

