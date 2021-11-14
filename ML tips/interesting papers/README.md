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

# Masked Autoencoders Are Scalable Vision Learners

paper: https://arxiv.org/abs/2111.06377

video explanation: https://www.youtube.com/watch?v=LKixq2S2Pz8

