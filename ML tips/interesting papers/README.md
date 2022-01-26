# CLIP
https://www.casualganpapers.com/zero-shot-contrastive-loss-image-text-pretraining/CLIP-explained.html

Github repo: https://github.com/openai/CLIP

## Notes
- `n` image - text pairs (`n` refers to a batch of images)
- `I_f = image_encoder(I)` has shape `[n, d_i]` where `n` is batch size `d_i` is output dimension of the encoder (e.g. 512, etc.)
- `T_f = text_encoder(T)` has shape `[n, d_t]` where `d_t` is output dimension of the encoder (512, 768 etc.)
- image encoder has an embedding matrix `W_i` of shape `[d_i, d_e]` where `d_e` is dimension (col) of the embedding matrix
  - output `I_e = l2_norm(np.dot(I_f, W_i), axis=1)` has shape `[n, d_e]`
- similar, text encoder has embedding matrix `W_t` of shape `[d_t, d_e]`
  - output `T_e = l2_norm(np.dot(T_f, W_t), axis=1)` has shape `[n, d_e]`

- final scaled logits = `np.dot(I_e, T_e.T)*np.exp(t)` `t` is a learned temperature param
  - output is shape `[n, n]`
  
- loss func 
  - target matrix should be an identity matrix of shape `[n, n]` (1 on diag, 0 everywhere else)
  ```
  labels = np.arrange(n)# = [0, 1, ..., n-1]
  loss_i = cross_entropy(logits, labels, axis=0) # (image is columns)
  loss_t = cross_entropy(logits, labels, axis=1) # (text is rows)
  loss = (loss_i + loss_j)/2
  ```
  - then can just to `loss.backward()` in the training
Note: in Openai implementation, the forward is 
```
def forward(self, image, text):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text
```
where `logits_per_image` is equivalent to `logits = np.dot(I_e, T_e.T)*np.exp(t)` and `logits_per_text` is equivalent to `logits.T`

# Towards Zero-Label Language Learning (Training model on synthetic data)
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

# IR related papers

## COIL 
https://arxiv.org/abs/2104.07186

- combines sparse and dense representations (think BM25 + bi-encoder)
- better than DPR, not that much slower

## Condenser
https://arxiv.org/abs/2104.08253

- a pre-training scheme 

## RDR
https://arxiv.org/abs/2010.10999

## Cross-Architecture Knowledge Distillation
github: https://github.com/sebastian-hofstaetter/neural-ranking-kd

source code for training models: https://github.com/sebastian-hofstaetter/matchmaker

paper: https://arxiv.org/abs/2010.02666

premise: train cross-encoder on QA data (performs much better, but slow), then distill (with MarginMSE) to a student bi-encoder model

## RocketQAv2
paper: https://arxiv.org/abs/2110.07367

github: https://github.com/PaddlePaddle/RocketQA

- equal or slightly worse than `coCondenser`

## coCondenser

paper: https://arxiv.org/abs/2108.05540

github: https://github.com/luyug/Condenser

- currently SOTA

## YONO
paper: https://arxiv.org/abs/2112.07381

github: currently none

- competitive/slightly better performance on ODQA (NQ/TriviaQA) using just retriever
- no results on MS-MARCO
- better performance when adding Ranker as well

## Colbert v2

paper: https://arxiv.org/abs/2112.01488

github https://github.com/stanford-futuredata/ColBERT/tree/new_api

slightly better than coCondenser on MS-MARCO, no results in NQ/TriviaQA

## GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval
paper: https://arxiv.org/abs/2112.07577

- used for unsupervised domain adaptation for dense retrieval
- no labelled data

# Masked Autoencoders Are Scalable Vision Learners

paper: https://arxiv.org/abs/2111.06377

video explanation: https://www.youtube.com/watch?v=LKixq2S2Pz8

# Efficiently and effectively scaling up language model pretraining for best language representation model on GLUE and SuperGLUE
https://www.microsoft.com/en-us/research/blog/efficiently-and-effectively-scaling-up-language-model-pretraining-for-best-language-representation-model-on-glue-and-superglue/?OCID=msr_blog_TNLRV5_tw

- no paper yet

# Deberta V3
paper: https://arxiv.org/abs/2111.09543

- replace MLM pretraining with replaced token detection (RTD) electra style

# MoE papers

MoE survey notes: https://hackmd.io/Ah9cd9fRR2uJLnvKExMkEw

## Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
paper: https://arxiv.org/abs/2101.03961

The first in a series of MoE papers

Main takeaways
- fine-tuning regularization: use dropout + expert dropout (dropping entire experts)
  - smaller models can outperform flop-matched dense models in downstream tasks for supervised fine-tuning
- various strategies to train large MoE models
![image](https://user-images.githubusercontent.com/8890262/150568439-262d8ad1-99b8-420c-8659-2c946edad005.png)

scaling MoE models to 1T+ parameters
- lm performance (neg log perplexity) improves, but downstream performance (such as SGLUE) does not outperform dense models (T5-XXL at 13B)

## GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
paper: https://arxiv.org/abs/2112.06905

summary of paper: https://twitter.com/MosaicML/status/1484253206112780291

- MoE LM
- more energy efficient training
- faster inference
- each input is fed through a gating layer to determine which of the 2 experts will be used 

![image](https://user-images.githubusercontent.com/8890262/150565245-958a192b-2ea0-4931-a966-efd516cf030a.png)


Dealing with training instability:
- debugging everything with smaller-scale models first
- skip updates with Nan/inf values
- restarting from an earlier checkpoint when there are “large fluctuations”

## DeepSpeed-MoE
https://deepai.org/publication/deepspeed-moe-advancing-mixture-of-experts-inference-and-training-to-power-next-generation-ai-scale

- deepspeed + MoE
- deepspeed MoE tutorials
  - https://www.deepspeed.ai/tutorials/mixture-of-experts/
  - https://www.deepspeed.ai/tutorials/mixture-of-experts-nlg/

## Efficient Large Scale Language Modeling with Mixtures of Experts
https://arxiv.org/pdf/2112.10684.pdf

Main findings:
- MoE models can indeed achieve similar downstream task performance as dense models at a fraction of the compute
- MoE models can yield competitive zero/few shot performance (compared to dense models) at fraction of computation for training and inference
- In a fully supervised fine-tuning regime, MoE models still underperform relative to dense models
  - Hypothesize MoE models may need alternative fine-tuning strategies compared to dense models



# Retro Retrieval-Enhanced TRansfOrmer

paper: https://arxiv.org/abs/2112.04426

http://jalammar.github.io/illustrated-retrieval-transformer/

- major benefit of retro: easier to scale a database than scaling model
  - model scaling is much more expensive
  - allows incorporating more knowledge than what can be encoded in model params
  - also allows scaling in 2 ways (db and model) as opposed to model only

question: can retreival enhance other tasks beside LM/IR/QA? eg classification
  - most naive version is to do a nearest neighbor search on database of vector representations with vector-label key-value pairing
  - can you use retrieval to augment classifier/encoder?
  - maybe better done as a seq2seq like t5/retro

3 possible ways to create a classifier
1. Have a classifier on top of the original model + nearest neighbors
  - train classifier, generate encoded representation (ie use layer before classifier layer's output), and store in DB along with true class
  - inputs to the final classifier: predicted probas of model, class of k nearest neighbors, l2 distance of each neighbor to model output vector
  - since the encoder is used to generate encoded representation AND classification, maybe using label smoothing might not be useful
    - should test with and without label smoothing
2. Use a retro style architecture, inject external vectors into model
  - requires 2 NNs, one (frozen) to generate the encoded representations and 1 to take (input + nearest neighbors) into the model
  - the encoder model should not use label smoothing in this case
3. frame classification as a seq2seq task (think t5), then use retro directly
