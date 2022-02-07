# Jax/Flax Notes

Notes for using Jax/Flax

## Flax basics
https://colab.research.google.com/github/BertrandRdp/flax/blob/master/docs/notebooks/flax_basics.ipynb

### Saving model
either use serialization: https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html#Serializing-the-result

or checkpoints https://flax.readthedocs.io/en/latest/flax.training.html

### Dealing with setting `training = True/False`
https://flax.readthedocs.io/en/latest/design_notes/arguments.html

add argument in `model.__call__` method eg 
```
from functools import partial
from flax import linen as nn

class ResidualModel(nn.Module):
  drop_rate: float

  @nn.compact
  def __call__(self, x, *, train):
    dropout = partial(nn.Dropout, rate=self.drop_rate, deterministic=not train) # <- setting train sets wether to use dropout or not
    for i in range(10):
      x += ResidualBlock(dropout=dropout, ...)(x)
```

### Mixed precision (float16/bfloat16) training
- set `model_detype` and `input_dtype` to float16
- scaling done using `DynamicScale`: https://flax.readthedocs.io/en/latest/_autosummary/flax.optim.DynamicScale.html
- full example of using float16 w/ dynamic scaling on imagenet here: https://github.com/google/flax/blob/main/examples/imagenet/train.py

### JAX/Flax on (multiple) GPUs

### model paralellism in JAX
done using `xmap` and `mesh` or `pjit`

xmap tutorial: https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html

mesh tutorial: https://jax.readthedocs.io/en/latest/jax.experimental.maps.html

mesh transformers (GPT-J6B) done using JAX and Haiku: https://github.com/kingoflolz/mesh-transformer-jax

### Training Flax/Haiku models using `optax`
- has `AdamW` and `OneCycleLR` (among other optimizers + lr schedules)
- distributed shampoo in optax: https://github.com/google-research/google-research/tree/master/scalable_shampoo
  - example usage of distributed shampoo vs Adam/AdaFactor: https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy
  - example training script with optax and distributed shampoo: https://github.com/borisdayma/dalle-mini/blob/main/tools/train/train.py

### `pjit` in JAX
- used for 2D paralellism (data + model parallelism)
- https://twitter.com/borisdayma/status/1486085583764135938?t=z_nsJ2ttUPExlykSQHwcIQ&s=09&fbclid=IwAR1lO3zaNWzLI82zKzHQ5wm9nMtQSTPH2_2-XgxrlLjjncNGLw1knL88HaE
  - example of `pjit` https://github.com/borisdayma/dalle-mini/commit/2b7f5f1daad2e3a24e883748ec3e818af5aab3b0
- `pjit` tutorial: https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html?fbclid=IwAR0nDEmjv1mOUY85qTrJEofyizYRAUZDqVGIbmTEJoQJEzwDd_VN80cHckE
- creating a 2D mesh for MP + DP in dalle_mini: https://github.com/borisdayma/dalle-mini/blob/main/tools/train/train.py

### persistent compilation cache
https://github.com/google/jax/issues/476#issuecomment-1015773039
```
from jax.experimental.compilation_cache import compilation_cache as cc

cc.initialize_cache("/path/name/here", max_cache_size_bytes=32 * 2**30)
```
- should speed up JAX operations
- only on TPUs (for now)

## kaggle tutorial notebooks
- https://www.kaggle.com/heyytanay/sentiment-clf-jax-flax-on-tpus-w-b?fbclid=IwAR3efsSkQfYxfncQhhT7yWlFs1L8BSkRx1TAfW_sHBM4xUB4Yu0CnNPWga8
- https://www.kaggle.com/asvskartheek/bert-tpus-jax-huggingface?fbclid=IwAR19964oeK47rhzlYxOV9o6esoz6VGH1JIgmAlRefOyUEBfDYcR1DH_wVKw
- https://www.kaggle.com/heyytanay/sentiment-clf-jax-flax-on-tpus-w-b

- uses `optax` for AdamW and OneCycleLR


## Distributed shampoo optimizer
https://github.com/google-research/google-research/tree/master/scalable_shampoo

- use the optax version of shampoo

- Distributed shampoo notes: https://twitter.com/borisdayma/status/1483845589280382976
  - use quantization on everything except diagonal statistics (should be the case by default)
  - quantization especially useful in single GPU/TPU settings
  - may not be necessary in multi-GPU/TPU setting due to sharding

- example of training model in JAX with distributed shampoo: https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy
## Huggingface JAX/Flax tutorials

### Pretraining Causal LM on TPU w/ JAX/Flax
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/causal_language_modeling_flax.ipynb

### Fine-tuning BERT on GLUE 
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=kTCFado4IrIc

## Other useful JAX projects/tutorials
Dalle Mini: https://github.com/borisdayma/dalle-mini
  - 2D parallelism (model parallel + data parallel) with mesh and `pjit`
  - uses distributed shampoo optimizer

