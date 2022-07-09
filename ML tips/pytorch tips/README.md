# pytorch specific tips

## general tips
https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/

- don't forget to call `optimizer.zero_grad()` before calling `loss.backward()` and `optmizer.step()`

more pytorch tips: https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/

Huggingface - [Performance and Scalability: How To Fit a Bigger Model and Train It Faster](https://huggingface.co/docs/transformers/performance)

Huggingface - [Model parallelism guide](https://huggingface.co/docs/transformers/parallelism#model-parallelism)

Here is a very rough outline at which parallelism strategy to use when. The first on each list is typically faster.

⇨ Single GPU

    Model fits onto a single GPU:
        Normal use

    Model doesn’t fit onto a single GPU:
        ZeRO + Offload CPU and optionally NVMe
        as above plus Memory Centric Tiling (see below for details) if the largest layer can’t fit into a single GPU

    Largest Layer not fitting into a single GPU:

    ZeRO - Enable Memory Centric Tiling (MCT). It allows you to run arbitrarily large layers by automatically splitting them and executing them sequentially. MCT reduces the number of parameters that are live on a GPU, but it does not affect the activation memory. As this need is very rare as of this writing a manual override of torch.nn.Linear needs to be done by the user.

⇨ Single Node / Multi-GPU

    Model fits onto a single GPU:
        DDP - Distributed DP
        ZeRO - may or may not be faster depending on the situation and configuration used

    Model doesn’t fit onto a single GPU:

        PP

        ZeRO

        TP

        With very fast intra-node connectivity of NVLINK or NVSwitch all three should be mostly on par, without these PP will be faster than TP or ZeRO. The degree of TP may also make a difference. Best to experiment to find the winner on your particular setup.

        TP is almost always used within a single node. That is TP size <= gpus per node.

    Largest Layer not fitting into a single GPU:
        If not using ZeRO - must use TP, as PP alone won’t be able to fit.
        With ZeRO see the same entry for “Single GPU” above

⇨ Multi-Node / Multi-GPU

    When you have fast inter-node connectivity:
        ZeRO - as it requires close to no modifications to the model
        PP+TP+DP - less communications, but requires massive changes to the model

    when you have slow inter-node connectivity and still low on GPU memory:
        DP+PP+TP+ZeRO-1

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
- reduce lr on plateau callback: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
- terminate on nan
- skipping update steps with `nan/inf`, may be better than terminating training entirely

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

## MoE training with DeepSpeed
1. MoE tutorial on cifar10: https://www.deepspeed.ai/tutorials/mixture-of-experts/
2. Applying MoE to Megatron-LM (gpt3 style autoregressive LM) for NLG: https://www.deepspeed.ai/tutorials/mixture-of-experts-nlg/


## Distributed training with Bagua
https://github.com/BaguaSys/bagua

examples: https://github.com/BaguaSys/bagua/tree/master/examples

## Dealing with numerical instability with mixed precision
- especially prevalent when scaling
- try using `"bf16"` instead of `16` for precision
  - improved numerical stability
  - useful for Ampere gpus (3090, A100 etc.)
  - requires pytorch `1.10.0` and up
- warning: if a model is pre-trained in `bf16`, fine-tuning in `fp16` will result in numerical instability
- bf16 is less precise than fp16
- no longer need to manually scale loss. especially useful if running mixed precision with `SAM`, 
```
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    loss, outputs = ...
```

## MosaicML tips to improve training

https://www.mosaicml.com/blog/5-best-practices-for-efficient-model-training

Algorithms that can improve training performance: https://docs.mosaicml.com/en/latest/trainer/algorithms.html



## Training recipe for LLMs (100B model)
https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6
- useful tricks for speeding up and stablizing training of large language models

# Huggingface Accelerate
https://github.com/huggingface/accelerate

## Dealing with gradient accumulation 
- need to deal with gradient accumulation manually in accelerate
- https://twitter.com/TheZachMueller/status/1541396521668575232?t=KtJspZqH7Pz3smxzlwgkMg&s=09&fbclid=IwAR2Zjo89ss9RRuCJXWz5evZqT8n3qeNrd9X-6iK9AHewHSWbYk8IT4nwu1Q
- error in the code posted, should be when `step % grad_accum_steps != 0: with accelerator.no_sync(model)...`

## Deepspeed in accelerate
- https://github.com/huggingface/accelerate#launching-training-using-deepspeed
- need to handle gradient accumulation manually
