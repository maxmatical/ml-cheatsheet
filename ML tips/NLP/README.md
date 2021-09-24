
# NLP:

## Fastai2 with transformers:
https://github.com/aikindergarten/fasthugs

https://github.com/ohmeow/blurr

**Tip:** seems like unfreezing and fine-tuning entire model can have equal or better performance than freezing and gradually unfreezing model

**General models to use**
- classification: (distil)roberta, bart, deberta

## **Label Smoothing**:

loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP


## Fixing mismatch between vocab size in data_clas and data_lm:

```
data_clas.vocab.itos = data_lm.vocab.itos

```
Before the following line
```
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
```
This has fixed the error.

## **Concatenating models to use metadata:**

https://towardsdatascience.com/next-best-action-prediction-with-text-and-metadata-building-an-agent-assistant-81117730be6b

and notebook https://www.kaggle.com/adai183/metadata-enhanced-text-classification

## **Can try using QRNN***

https://github.com/piegu/language-models 

https://github.com/piegu/language-models/blob/master/lm2-french.ipynb 


## **AWD-LSTM Specific**

- Higher drop_mult on LM learner (1.), then smaller dropout on classifier

- Ensemble a fwd and backwards bwd = True models
  - backwards model: https://github.com/fastai/course-nlp/blob/master/nn-vietnamese-bwd.ipynb 
  - ensemble model: https://github.com/fastai/course-nlp 
  
## Data augmentation for NLP
- backtranslation - translate text into another language, then translate back: https://amitness.com/2020/02/back-translation-in-google-sheets/
- other methods: https://arxiv.org/abs/2106.07499
- data augmentation library https://github.com/facebookresearch/AugLy

### repo with transformer related code (training, inference, hpo, pseudolabel training)
https://github.com/maxmatical/fastai-transformers

## GPT-x for zero-shot learning

GPT-J-6B (on par with GPT-3 6.7B model): https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/

example notebook: https://github.com/maxmatical/fast.ai/blob/master/GPT_J_6B_Topic_Modelling.ipynb

GPT-J-6B on gpus: https://github.com/paulcjh/gpt-j-6b/blob/main/gpt-j-t4.ipynb

Fine-tune GPT-J-6B (requires TPUs): https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md

GPT-J on pytorch vis huggingface (make sure using half-precision): https://huggingface.co/transformers/master/model_doc/gptj.html

## Training model on synthetic data
- https://arxiv.org/pdf/2109.09193.pdf
- use a large language model (T5, GPT3, GPTJ, etc.)
- use prompt to generate sample text **conditioned** on a label, eg
  ```
  Sample Movie Review: This is the most saccharine romance I ever
  sat through. The perfect film for an idle housewife in kerchief,
  housedress, and ostrich-trimmed high-heeled mules to watch in the
  afternoon, lying on the couch eating bonbons. In fact, bonbons play a
  prominent role in the movie. The only reason I was able to watch to
  the end, is that I finally was able to gaze at Keanu Reeves’ dreamy
  face in almost every scene. In most of his films, he moves too fast
  to get a good look. The only rapid action in this show is Giancarlo
  Giannini waving his hands with Latin emotionality - more Italian than
  Mexican, really.
  
  Negative Movie Review:
  ```
- train discriminative model (BERT, T5, BART, etc.) using label and synthetic text
- (Optional?) fine-tune on actual dataset

## Topic Modelling
- unsupervised topic mining
- https://hackmd.io/uVFpqWb9Q0KV3fmq0LdvMA

### Contextualized Topic Models (CTM)
- SOTA Survey: https://silviatti.github.io/resources/alliancebernstein_30_10_20.pdf
- [Cross-lingual Contextualized Topic Models with Zero-shot Learning](https://paperswithcode.com/paper/cross-lingual-contextualized-topic-models)
  - contexualized topic models (CTM) github repo: https://github.com/MilaNLProc/contextualized-topic-models
  - Example tutorial with CTM: https://colab.research.google.com/drive/1fXJjr_rwqvpp1IdNQ4dxqN4Dp88cxO97?usp=sharing#scrollTo=iZEPr_QFJdBz

## Bi-Encoder for Neural Search
https://www.youtube.com/watch?v=XHY-3FzaLGc

![training process](https://github.com/maxmatical/ml-cheatsheet/blob/master/images/neural_search.png)

Sapmle code: https://www.sbert.net/examples/training/ms_marco/README.html#ms-marco

used for 
- search
- Information retreival
- Question Answering
