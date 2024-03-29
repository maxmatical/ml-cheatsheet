# Compiliing list of SOTA optimizers

See `optimizers.py`

## Summary of optimizers to try
See this thread: https://twitter.com/borisdayma/status/1479068659146534917?t=a79kxQ2c_oU_BFnkl7hyyw&s=09&fbclid=IwAR0xaBcED36-4ONnq2DD2gi5i4NUdDyLCY9zciiHrufVHTapbaA9Fgzc0Yo

For performance
- SGDM
- RMSPROP
- LAMB
- AdamW
- Ranger
- Ranger21
- RangerAdabelief
- distributed shampoo
- (Adaptive)/(Efficient) SAM + optimizer
- [Adan](https://twitter.com/davisblalock/status/1561976182567870465)
- Amos

For training large models when memory is an issue
- AdaFactor (although AdamW might be better for Large LMs, see twitter thread and GOPHER paper)
- Distributed Shampoo 
  - more memory overhead than adam, but more stable, better performance for large models
- Novograd
- AdaGraft
- SM3
- Try 8 bit optimizers (eg 8bit adamw) https://github.com/facebookresearch/bitsandbytes
- hybridadam for colossalai
- LION


## Ranger seems to work really well (try with both `fit_one_cycle` and `fit_flat_cos`)
  - `fit_flat_cos` seems to work better for Ranger
![ranger](https://github.com/maxmatical/fast.ai/blob/master/ranger.png)

https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

https://github.com/mgrankin/over9000

For adam based optimizers in general, try setting eps =[1.0, 0.1, 0.01]. Change with
```
eps = 1e-4
learn.opt_func = partial(learn.opt_func, eps=eps)
```
## [RangerAdabelief](https://forums.fast.ai/t/gradient-centralization-ranger-optimizer-updated-with-it/68420/18)

[RangerAdabelief repo](https://github.com/juntang-zhuang/Adabelief-Optimizer)

[RangerAdabelief episilon values](https://github.com/juntang-zhuang/Adabelief-Optimizer#2-epsilon)

For **RangerAdabelief**, try `eps=1e-8` for CV, `eps=1e-16` for NLP, `eps=1e-12` for RL as default values (try a eps sweep for hyperparam tuning)

- **set `true_wd = True, bn_wd = False`**

- **Note:** run `learn.fit_flat_cos()` with new optimizers (flat + cosine anneling)

According to https://www.reddit.com/r/MachineLearning/comments/dhws0l/r_on_the_adequacy_of_untuned_warmup_for_adaptive/, it seems like AdamW may still be competitive


## Older optimizers (SGD, RMSPROP)
Try using **SGD (momentum = 0.9, nesterov = True) or RMSPROP(momentum=0.9)**, can maybe generalize better (try for CV, maybe also works for NLP)


## 8 bit optimizers
https://github.com/TimDettmers/bitsandbytes

saves up to 75% memory on training models

may require running embedding layers in fp32 (see this discussion for details: https://github.com/huggingface/transformers/issues/14819), eg
```
import torch
import bitsandbytes as bnb
from transformers import GPTNeoForCausalLM
from bitsandbytes.optim import GlobalOptimManager

def set_optim_to_run_embedding_in_fp32(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            GlobalOptimManager.get_instance().register_module_override(module, 'weight', {'optim_bits': 32})

mname = "EleutherAI/gpt-neo-125M"
model = GPTNeoForCausalLM.from_pretrained(mname)
set_optim_to_run_embedding_in_fp32(model)

```

## [Sharpness-Aware Minimization (SAM) optimizer](https://github.com/davda54/sam) 
- seems to be useful for CV and NLP
- May be especially good for `ViT` and `Mlp_Mixer`
- Good to prevent overfitting, no real benefit when each datapoint is only seen once (eg LLM training setup)
- values for `rho` to try: `{0.01, 0.02, 0.05, 0.1, 0.2, 0.5}`
- Using SAM in  https://github.com/davda54/sam/
  - Note: to use SAM with FP16 (AMP) https://github.com/davda54/sam/issues/7
  - Should likely not use FP16 due to issues
  - gradient clipping: https://github.com/davda54/sam/issues/64
  - using gradient accumulation:
    - https://github.com/maxmatical/ml-cheatsheet/blob/master/Pytorch_Lightning_BERT_Huggingface_w_SAM_%2B_EMA.ipynb
    - https://github.com/davda54/sam/issues/3
  - using SAM in pytorch lightning: https://github.com/davda54/sam/issues/42
- composer implementation of SAM: https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam

### SAM extensions:
- SAM in composer (might be better impl): https://github.com/mosaicml/composer/tree/dev/composer/algorithms/sam
- Efficient sam: https://arxiv.org/abs/2203.02714
- GSAM: https://arxiv.org/abs/2203.08065
  - Better performance than SAM
  - Github code: https://github.com/juntang-zhuang/GSAM
- ESAM: https://arxiv.org/abs/2205.14083
  - Faster than (G)SAM
  - Competitive results in imagenet w/ Resnets. Bit worse for ViTs
  - No code currently 
  
### SAM may be too slow
- https://twitter.com/rasbt/status/1626226500461793282
- stochastic weight perturbation (advasarial weight perturbation?) may be more efficient
  - same w/ SWA?

## [Ranger21](https://github.com/lessw2020/Ranger21)
- no need for lr schedule (define lr schedule in the optimizer itself), just need to call `trainer.fit()`, `learner.fit()`, etc.
- `use_madgrad = True` might be better for transformers

## Distributed Shampoo 
- pytorch: https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo
  - fsdp support is still experimental
  - who knows if it's actually better than adamw
- jax/optax: https://github.com/google-research/google-research/tree/master/scalable_shampoo
- possible exension: fishy: https://openreview.net/forum?id=cScb-RrBQC

## Adan optimizer
- https://twitter.com/davisblalock/status/1561976182567870465
- arxiv: https://arxiv.org/abs/2208.06677
- beats AdamW, SAM, SGD, Adabelief, etc. in Vision/NLP/RL tasks
- Code: https://github.com/sail-sg/Adan
- (non-official) pytorch implementation: https://github.com/lucidrains/Adan-pytorch
- report on Adan vs adam: https://wandb.ai/capecape/adan_optimizer/reports/Adan-A-New-Optimizer-That-Challenges-Adam--VmlldzoyNTQ5NjQ5
  - recommends `betas = (0.02, 0.08, 0.01)` aka baseline values
  - recommends 10x lr of adam
  - seems to perform very similar to AdamW, but slower
    - note: only on 1 model for image classification
  - need comparison to lookahead/Ranger21/rangeradabelief etc. to decide the "best" optimizer
- at 1B+ scale memory could become an issue
  - at 350M, 5% extra memory usage, at 1B+, 50% extra memory (from big code experiments)
  
  
## Amos optimizer
- https://twitter.com/Robin_Tian/status/1585311211457249280
- converges faster than AdamW, but also better performance
- up to 50% reduction in slot variables (eg running averages in AdamW), up to 70% reduction in training time (# of training steps)

## Some common optimizer configs
```
optimizer_config_mapping = {
    "adamw": {"optimizer": AdamW, "wd": {"true_wd": True, "bn_wd": True}},
    "ranger": {"optimizer": Ranger, "wd": {"true_wd": True, "bn_wd": False}},
    "ranger_nogc": {
        "optimizer": Ranger,
        "hyperparameters": {"use_gc": False},
        "wd": {"true_wd": True, "bn_wd": False},
    },
    "rangeradabelief": {"optimizer": RangerAdaBelief, "wd": {"true_wd": True, "bn_wd": False}},
    "rangeradabelief_nogc": {
        "optimizer": RangerAdaBelief,
        "hyperparameters": {"use_gc": False},
        "wd": {"true_wd": True, "bn_wd": False},
    },
    "sgd": {
        "optimizer": SGD,
        "hyperparameters": {"momentum": 0, "nesterov": False},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
    "sgd_momentum": {
        "optimizer": SGD,
        "hyperparameters": {"momentum": 0.9, "nesterov": False},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
    "sgd_nesterov": {
        "optimizer": SGD,
        "hyperparameters": {"momentum": 0.9, "nesterov": True},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
    "rmsprop": {
        "optimizer": RMSprop,
        "hyperparameters": {"momentum": 0.9},
        "wd": {"true_wd": True, "bn_wd": False},
        "set_betas": False,
    },
}
```

## AdamW no decay for bias/layernorm weight

Consider setting the following (for transformers, AdamW only?)
```
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
```

## Setting `foreach=True` in pytorch optimizers for fused optimizers
- torch >= 1.13
- should speedup optimizer
- in fastai: https://fastxtend.benjaminwarner.dev/optimizer.fused.html




## LION optimizer
- https://arxiv.org/abs/2302.06675
- https://twitter.com/DrJimFan/status/1625920773042089984
- more mem efficient than AdamW, reports better performance too in CV/NLP tasks
- may require:
  - larger batch size (larger = better, min >= 64)
  - smaller lr
  - larger wd (to get simiar effect of wd as adamw) if not using wd keep to 0 (which may be better in most cases)
- torch implementation: https://github.com/lucidrains/lion-pytorch
- still needs to be validated
  - https://twitter.com/rasbt/status/1633935266053251074
  - reports similar performance, but mem savings
- apparently doesn't work as well on RL?
- Mosaic MPT uses LION optimizer - reports mem savings and better stability
  - mosaic impl: https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/optim
    - includes an adaptive variant that automatically detects outliers in moment and decreases
  - note: mosaic example yamls doesn't contain the actual mpt 7b run with lion optimizer -> hyperparams not known

## Benchmarking Neural Network Training Algorithms
- arxiv: https://arxiv.org/abs/2306.07179
- tweet: https://twitter.com/arankomatsuzaki/status/1668421282449305602
- Looked at:
  - AdamW
  - NadamW
  - Distributed Shampoo
  - LAMB
  - SAM (w Adamw)
- NadamW seems to perform particularly well
- SAM/shampoo takes 2x time per step
- SAM doesn't do well




## No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models
- 3 classes of algorithms for efficient training tested on BERT and T5 pretraining
- dynamic architecture: layer stacking, layer dropping
- batch selection: selective backprop, RHO loss
- efficient optimizers: Lion, Sophia
- "When pre-training BERT and T5 with a fixed computation budget using such methods, we find that their training, validation, and downstream gains vanish compared to a baseline with a fully-decayed learning rate."
- all these fancy tricks to improve llm training, yet nothing beats plain adamw + more tokens. although the training setup is quite constrained in certain aspects

## Optimizer choices for finetuning
- SGD may be preferable for finetuning over adam
    - less memory, less tunable parameters, not much difference in results
    - try using a much lower LR (like 10x lower?)
    - may not be suited for pretraining though!
    - empirically found not to give good results, but still worth testing in mem constrained scenarios

- maybe the above also applies to lion?

- some finetuning tips: https://twitter.com/giffmana/status/1634828535729618944
    - low lr
    - no wd
    - zero init new linear head

- momentum *may* be needed to get higher quality models, see: https://twitter.com/_arohan_/status/1630062139259101185
