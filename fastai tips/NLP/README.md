
## NLP:

### Fastai2 with transformers:
https://github.com/aikindergarten/fasthugs

https://github.com/ohmeow/blurr

### data augmentation: back translate
https://amitness.com/2020/02/back-translation-in-google-sheets/

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

### Transformers (BERT, RoBERTa, etc.) with fastai
RoBERTa: https://medium.com/analytics-vidhya/using-roberta-with-fastai-for-nlp-7ed3fed21f6c

Huggingface transformers: https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2?fbclid=IwAR2_MQh2XzsKEKqwckIShH8-wO5io6rD1wXb4nTn3-eABj8boE9jOYd3zsM 

**Tip:** sometimes training the entire model can have equal or better performance than freezing and gradually unfreezing model

### **AWD-LSTM Specific**

- Higher drop_mult on LM learner (1.), then smaller dropout on classifier

- Ensemble a fwd and backwards bwd = True models
  - backwards model: https://github.com/fastai/course-nlp/blob/master/nn-vietnamese-bwd.ipynb 
  - ensemble model: https://github.com/fastai/course-nlp 
  
### Data augmentation for NLP
- translate text into another language, then translate back

### repo with transformer related code (training, inference, hpo, pseudolabel training)
https://github.com/maxmatical/fastai-transformers
