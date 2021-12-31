# pytorch specific tips

## general tips
https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/

- don't forget to call `optimizer.zero_grad()` before calling `loss.backward()` and `optmizer.step()`

more pytorch tips: https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/


## integrating fastai functions

### using fastai in pytorch training loop
[example notebook](https://github.com/maxmatical/ml-cheatsheet/blob/master/imagenette_with_pytorch.ipynb)
- uses fastai loss functions and `flat_cos` learning rate scheduler 

### learning rate 
- lr finder: https://github.com/davidtvs/pytorch-lr-finder
- fit_flat_cos lr schedule (https://github.com/mgrankin/over9000/blob/master/train.py)
- reduce lr on plateau (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau)
```
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min')
for epoch in range(10):
    train(...)
    val_loss = validate(...)
    # Note that step should be called after validate()
    scheduler.step(val_loss)
```


## multi-gpu training (distributed data parallel)
- distributed training on multiple gpus: https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch
- DDP tutorial (covers multi-node as well) https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html


## mix precision training (amp)
- fp16 (mixed precision training): https://pytorch.org/docs/stable/notes/amp_examples.html
    - combining amp with lr schedulers: https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930
    - example of amp with `OneCycleLR` https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam

## Multi gpu (DDP) with fp16 (amp)
https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus

- no real need to modify training loop, just use amp with DDP

## model parallel with deepspeed
- used for model parallelism (when model doesn't fit on 1 gpu)
- contains zero optimizer
- use DeepSpeed https://github.com/microsoft/DeepSpeed (model parallelism, ZeRO optimizer, etc.)
  - [getting started tutorial with pytorch module](https://www.deepspeed.ai/getting-started/)
- simple model parallelism https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html


## Callbacks
- early stopping
- save model
- reduce lr on plateau callback: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
- terminate on nan callback

## mixup
- mixup: https://towardsdatascience.com/enhancing-neural-networks-with-mixup-in-pytorch-5129d261bc4a
  - call be a callback?
- pytorch implementation: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py#L152
  - useful for CV, possibly for NLP in classification tasks

## SAM
- add SAM wrapper for optimizer and modify training `if self.sam`: https://github.com/davda54/sam
- SAM with gradient accumulation:https://github.com/davda54/sam/issues/3
- SAM with FP16: https://github.com/davda54/sam/issues/7


## loss function and metrics
- keep `FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1)` and `accuracy` from fastai


## Avoid reusing stateless modules
https://twitter.com/ThomasViehmann/status/1452199693165998081

- eg `nn.Dropout` or activations like `nn.ReLu`
- Use multiple instances instead of resuing, or use `torch.nn.functional` instead

## Batch size finder using koila
https://github.com/rentruewang/koila

## hard example mining/batch loss filter
pytorch loss func: https://erogol.com/online-hard-example-mining-pytorch/

idea: sort batch of x, y by loss(x, y), and only take the top k% of the batch by loss (so model only sees the hard examples to use for backprop)
