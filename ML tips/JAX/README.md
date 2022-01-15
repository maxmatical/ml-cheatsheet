# Jax/Flax Notes

Notes for using Jax/Flax

## Flax basics
https://colab.research.google.com/github/BertrandRdp/flax/blob/master/docs/notebooks/flax_basics.ipynb

### Saving best model
either use serialization: https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html#Serializing-the-result

or checkpoints https://flax.readthedocs.io/en/latest/flax.training.html

## kaggle tutorial notebooks
- https://www.kaggle.com/heyytanay/sentiment-clf-jax-flax-on-tpus-w-b?fbclid=IwAR3efsSkQfYxfncQhhT7yWlFs1L8BSkRx1TAfW_sHBM4xUB4Yu0CnNPWga8
- https://www.kaggle.com/asvskartheek/bert-tpus-jax-huggingface?fbclid=IwAR19964oeK47rhzlYxOV9o6esoz6VGH1JIgmAlRefOyUEBfDYcR1DH_wVKw

- uses `optax` for AdamW and OneCycleLR


## Distributed shampoo optimizer
https://github.com/google-research/google-research/tree/master/scalable_shampoo

## Huggingface JAX/Flax tutorials

### Pretraining Causal LM on TPU w/ JAX/Flax
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/causal_language_modeling_flax.ipynb

### Fine-tuning BERT on GLUE 
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=kTCFado4IrIc
