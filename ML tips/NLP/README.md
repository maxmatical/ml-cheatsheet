
# NLP:

## Fastai2 with transformers:
https://github.com/aikindergarten/fasthugs

https://github.com/ohmeow/blurr

**Tip:** seems like unfreezing and fine-tuning entire model can have equal or better performance than freezing and gradually unfreezing model

**General models to use**
- classification: (distil)roberta, bart, deberta

## **Label Smoothing**:

loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP

## SAM for NLP
https://github.com/SamsungLabs/ASAM?fbclid=IwAR0_-1leTZ_loGjIDM-y2q-N6jQPdQA0AApNMOCQA2FeVOe5yq0zvQHuRSk

[seems to work well for NLP as well](https://arxiv.org/abs/2110.08529)


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
- Methods:
  - LDA
  - NMF
  - SBERT + Clustering
  - GPT 
- https://hackmd.io/uVFpqWb9Q0KV3fmq0LdvMA

### Contextualized Topic Models (CTM)
- SOTA Survey: https://silviatti.github.io/resources/alliancebernstein_30_10_20.pdf
- [Cross-lingual Contextualized Topic Models with Zero-shot Learning](https://paperswithcode.com/paper/cross-lingual-contextualized-topic-models)
  - contexualized topic models (CTM) github repo: https://github.com/MilaNLProc/contextualized-topic-models
  - Example tutorial with CTM: https://colab.research.google.com/drive/1fXJjr_rwqvpp1IdNQ4dxqN4Dp88cxO97?usp=sharing#scrollTo=iZEPr_QFJdBz

**Notes**:
- the encoder generates a `mu` and `sigma`, which are of size `(bs, n_components)` aka `n_topics`
- to get a predicted topic, the decoder network samples `n` times from a gaussian distribution with `mu, sigma` to get `theta`, then take the argmax of the avg probabilities 
  ```
  def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
            #posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
  ```
  ![image](https://user-images.githubusercontent.com/8890262/136230216-35286ecf-2788-486a-b2a9-88fcc127a3bf.png)
  - here `|K| == n_components`
  - take argmax of sampled representation to get topic

- Possible extension: use GPT (or some summarization model) as a summarizer (reduce the length of text to make it easier to learn), then use CTM 
  - CTM is used for ~200 tokens, so maybe not as useful for super short documents
  - maybe inject GPT topic as another context vector (either another vector to concatenate, or add GPT topic to original text)

## Training State-of-the-art Text Embedding Models
video: https://www.youtube.com/watch?v=XHY-3FzaLGc

![training process](https://github.com/maxmatical/ml-cheatsheet/blob/master/images/neural_search.png)

Sample code: https://www.sbert.net/examples/training/ms_marco/README.html#ms-marco

used for 
- search
- Information retreival
- Question Answering

Uses a similarity matrix for loss (similar to CLIP)

### Information Retreival Pipeline
- bi-encoder (for large collections of docs): https://www.sbert.net/examples/training/ms_marco/README.html#bi-encoder 
  - train using `MultipleNegativesRankingLoss`, `MarginMSE`, or the method described above for SOTA text embedding model
- cross encoder (for smaller collections OR after candidate retreival from bi-encoder): https://www.sbert.net/examples/training/ms_marco/README.html#cross-encoder
- One way to combine both encoders: https://www.sbert.net/examples/applications/retrieve_rerank/README.html
- **combine cross-encoder and bi-encoder with `MarginMSE`**
    - train cross encoder (CE) first (either from scratch or knowledge distillation)
    - for each `(query, doc1, doc2)` triplet, calculate
        - `CE_distance = CEScore(query, doc1) - CEScore(query, doc2)`, (cache `CEScore` as `{query, doc, score}`)
            - alternatively: use `CE` in the training. don't add `CE` params to optimizer, freeze parameters, and make sure gradients aren't tracked when calling `CE` (either `no_grad` or `inference_mode`) to compute `CEScore`
        - `BE_distance = BEScore(query, doc1) - BEScore(query, doc2)` (by default `BEScore` is dot-product, but can also be cosine-similarity)


### IR/Neural Search for low resource scenarios
https://twitter.com/Nils_Reimers/status/1452931505995624451?t=KYJjn4zKvjxaeCtnZgHWPg&s=09&fbclid=IwAR1dp1_QLuPabCnTF4lxWw8LSzVm5IsdGEPomhFYu7J5zjFqlA2_BeXKdlA

video: https://www.youtube.com/watch?v=XNJThigyvos

IR models on zero shot BEIR, ranked by performance
<img width="1375" alt="image" src="https://user-images.githubusercontent.com/8890262/138903081-2608258f-6688-4c37-ab5b-6083306aada5.png">

Interesting upcoming work to keep track of: GPL for Domain Adaptation
  - Generate queries for docs in your domain
  - Fine-tune bi-encoder
  - Improves performances 4 – 10 points

## Mixup for text
twitter thread: https://twitter.com/TheZachMueller/status/1451187672072921101

paper: https://arxiv.org/abs/2010.02394
- use mixup implementation https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py#L152
    - use mixup in the `forward`, and `return (out, targets_a, targets_b) if self.train else out`
