
## NLP:

### Fastai2 with transformers:
https://github.com/aikindergarten/fasthugs

https://github.com/ohmeow/blurr

**Tip:** seems like unfreezing and fine-tuning entire model can have equal or better performance than freezing and gradually unfreezing model

### **Label Smoothing**:

loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP


### Fixing mismatch between vocab size in data_clas and data_lm:

```
data_clas.vocab.itos = data_lm.vocab.itos

```
Before the following line
```
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
```
This has fixed the error.

### **Concatenating models to use metadata:**

https://towardsdatascience.com/next-best-action-prediction-with-text-and-metadata-building-an-agent-assistant-81117730be6b

and notebook https://www.kaggle.com/adai183/metadata-enhanced-text-classification

### **Can try using QRNN***

https://github.com/piegu/language-models 

https://github.com/piegu/language-models/blob/master/lm2-french.ipynb 


### **AWD-LSTM Specific**

- Higher drop_mult on LM learner (1.), then smaller dropout on classifier

- Ensemble a fwd and backwards bwd = True models
  - backwards model: https://github.com/fastai/course-nlp/blob/master/nn-vietnamese-bwd.ipynb 
  - ensemble model: https://github.com/fastai/course-nlp 
  
### Data augmentation for NLP
- backtranslation - translate text into another language, then translate back: https://amitness.com/2020/02/back-translation-in-google-sheets/
- other methods: https://arxiv.org/abs/2106.07499

### repo with transformer related code (training, inference, hpo, pseudolabel training)
https://github.com/maxmatical/fastai-transformers

### GPT-x for zero-shot learning

GPT-J-6B (on par with GPT-3 6.7B model): https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/

example notebook: https://github.com/maxmatical/fast.ai/blob/master/GPT_J_6B_Topic_Modelling.ipynb
