
# NLP:

## Interesting Kaggle NLP challenge solutions

[NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/323085)
- 2nd place solution
- deberta model ensemble
- training data fairly small
- Use classifier head w/ 5 different levels of dropout and averaged 5 logits

[We are all alike, on the inside](https://www.kaggle.com/c/we-are-all-alike-on-the-inside/discussion/312371)
- 1st place solution
- Use classifier head w/ 5 different levels of dropout and averaged 5 logits


[chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287923?fbclid=IwAR2unJF_zFq0wGTU8d4h_FqcA9JnVrGieriOgVigQoTrWNXnADGYE-E0I1M)
- 1st place solution
- interesting data augmentation techniques inspired by fastai/CV data augmentations

[US patent phrase to phrase matching](https://twitter.com/marktenenholtz/status/1539578965920083968?t=FEZfQssOLCZAFfQs3Mtu5g&s=09&fbclid=IwAR3yePeplbJAjyjMhl9EF-Pjuoph9IqIC2dKWHriFF4zFKVukXouegsjd8k)
- solo silver writeup
- feature engineering for text by clever combination of text e.g. `"similarity of {anchor} to {target} in relation to {CPC description}"`
- 8-bit Adam for larger batch sizes
- linear optimizer for ensembling over stacking/boosting
- More solutions
  - [1st place solution](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)
  - [2nd place solution](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)

[Feedback Prize - Predicting Effective Arguments 1st place solution](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)


[Feedback Prize - Predicting Effective Arguments 2nd place solution](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx#full-data-and-stochastic-weight-averaging-(swa))
- domain adaptation by pretraining on target corpus
- uses SWA to improve generalization performance
- uses [Adversarial Weight Perturbation (AWP)](https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook)
  - required more hyperparam tuning + longer training
  
## Kaggle - how to build an efficient NLP model 
based off of [Feedback Prize - Predicting Effective Arguments 2nd place solution](https://wandb.ai/darek/fbck/reports/How-To-Build-an-Efficient-NLP-Model--VmlldzoyNTE5MDEx)

### deberta
- most top solutions for nlp converged to deberta for seq classification tasks
- if > 512 tokens, longformer works better than bert/roberta/bart, but deberta can handle >512 tokens
- note for deberta >v2 use new vocab
- encoder-decoder models may work better for seq2seq tasks (translation, QA, summarization)
    - classification -> better to use encoder only
    - T5 would be an interesting model to try as well

### data augmentation for NLP
- random crop: take only a part of text
- cutout: replace tokens with `MASK` token -> very common technique in kaggle comps (also called token dropout, random masking)
- progressive resizing: start with shorter seq len -> grow as you train

### advarsarial weight perturbation (AWP)
- during training, perturb the inputs and weights
- leads to flatter loss landscape -> better generalization 
- requires longer training (2-3x)
- requires tuining `adv_lr`, `adv_eps`, `start_epoch`
- check the AWP notebook on kaggle

### using additional data (pseudolabelling)
- might have acess to a lot of data, but unlabelled
    - this was the case with this kaggle comp
    - had a lot of essays form prev comps that didn't have labels
1. additional mlm in-domain pre-training (domain adaptation)
2. multi-task (multi-stage) finetuning
    - finetune on auxillary task(a similar task) then finetune on target task (sequential)
3. pseudolabels
    - train on labelled data (1 or multiple models)
    - generate pseudolabels for unlabelled data
    - noisier labels, but lots more data
    - soft labels work better than hard labels in practice
        - take all data instead of just high confidence hard labels
        - use KL divergence loss on soft labels! (check logits or confidence?)
            - seems like predicted probability
    - first train on pseudolabeled data ONLY via KL divergence, then finetune on labeled data normally

### additional techniques
1. setting dropout to 0
    - all layers in transformers
    - may vary across arch/tasks
        - may need to experiment (may be case by case basis)
2. SWA
    - average weights across checkpoints
    - took task 3 checkpoints instead of a % of training epochs


## Fastai2 with transformers:
https://github.com/ohmeow/blurr

**Tip:** seems like unfreezing and fine-tuning entire model can have equal or better performance than freezing and gradually unfreezing model

## General models to use
- classification: (distil)roberta, bart, deberta ([deberta-v3](https://huggingface.co/microsoft/deberta-v3-small)), electra, t5 (conditional generation)

## **Label Smoothing**:

loss_func=FlattenedLoss(LabelSmoothingCrossEntropy, axis=-1) for NLP

## SAM for NLP
https://github.com/davda54/sam

[seems to work well for NLP as well](https://arxiv.org/abs/2110.08529)


## Data augmentation for NLP
- backtranslation - translate text into another language, then translate back: https://amitness.com/2020/02/back-translation-in-google-sheets/
- other methods: https://arxiv.org/abs/2106.07499
- data augmentation library https://github.com/facebookresearch/AugLy

### additional data augmentation techniques (from kaggle winning solution)
https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/287923?fbclid=IwAR2unJF_zFq0wGTU8d4h_FqcA9JnVrGieriOgVigQoTrWNXnADGYE-E0I1M

- random cropping
- progressive resizing
  - eg start with `max_length` in tokenizer of `128`, then `256`, etc.
- cutout: replace 0-10% of tokens with `[MASK]`
  - this will need to be done in the collator function

### NL Augmenter
https://github.com/GEM-benchmark/NL-Augmenter

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
- (Optional) fine-tune on actual dataset, can use dev set performane to decide whether or not to perform this step

### synthetic data in IR
InPars:
- https://arxiv.org/abs/2202.05144
- Use GPT3 to generate questions from a document (simlar to GPL)
- Fine-tune on synethetic + real data

GPL: using finetuned T5 to generate queries from a document


## Topic Modelling
- unsupervised topic mining
- Methods:
  - LDA
  - NMF
  - SBERT + Clustering
  - GPT 
- https://hackmd.io/uVFpqWb9Q0KV3fmq0LdvMA

### Distilling sentences to make it easier to model topics
- intuition: shorter sentences lead -> easier to cluster -> better topic model topics
- zero-shot: using GPT/LLM model for zero-shot summarization using prompt engineering
- supervised: use a model like BART to distill text into shorter text

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
  - instead of reconstructing original sentence, construct a distilled sentence (eg summarized by GPT/BART etc.)


- idea: 
  - embeddings seems to work better on shorter sentences? (from observation)
  - use gpt to "summarize"
  - use trained/pretrained autoencoder to generate embeddings
  - cluster/knn from that?

### using BERTopic for topic modelling
https://www.pinecone.io/learn/bertopic/?utm_content=207629974&utm_medium=social&utm_source=linkedin&hss_channel=lcp-20299330

- transformer embedding model
- UMAP dimensionality reduction
- HDBSCAN clustering
- Cluster tagging using c-TF-IDF

use custom tranformer model with
```
topic_model = BERTopic(embedding_model={$hf-model-checkpoint})
```

### BertTopic + GPU with cuML
https://medium.com/bumble-tech/multilingual-gpu-powered-topic-modelling-at-scale-dc8bd08609ef?fbclid=IwAR05S4dwVZcAvpJ3e7jks2yaUTJ_6LA2ONS67iayBU_OX6VV2GF5j0FbW_8

## Mixup for text
twitter thread: https://twitter.com/TheZachMueller/status/1451187672072921101

paper: https://arxiv.org/abs/2010.02394
- use mixup implementation https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py#L152
    - use mixup in the `forward`, and `return (out, targets_a, targets_b) if self.train else out`


## improving NLP models without scaling
https://hackmd.io/0n1aijQJRMy9xy2oHeQEXg


## GPT-NeoX 20B
https://github.com/EleutherAI/gpt-neox
- can use for zero-shot learning with prompt engineering

paper: http://eaidata.bmk.sh/data/GPT_NeoX_20B.pdf

running GPT-NeoX inference with T4s: https://nlpcloud.io/deploying-gpt-neox-20-production-focus-deepspeed.html?utm_source=reddit&utm_campaign=ehyiek56-ed8e-11eb-ba80-5242ac13d5jv
  - useful for topic modelling/distilling longer sentences into shorter ones (see topic modelling)


## domain adaptation of text embedding models

https://www.sbert.net/examples/domain_adaptation/README.html
- eg adapt a SBERT model to your text corpus
- data can come in different formats see: https://huggingface.co/datasets/sentence-transformers/embedding-training-data
  - can have a similar set up to DPR with `query, pos` pairs


## replace attention layers with ALIBI in decoder only LMs
https://docs.mosaicml.com/en/v0.5.0/method_cards/alibi.html

## Mosaic ML: Composer - ALiBi for BERT/RoBERTa
https://docs.mosaicml.com/en/stable/method_cards/alibi.html#alibi
- requires composer >= 0.9
- Supports BERT/RoBERTa and similar archs
  - will log an error if model class is unsupported (double check)

## Self training for NLP (few-shot classification)

STraTA: Self-Training with Task Augmentation for Better Few-shot Learning
  - https://arxiv.org/abs/2109.06270
  - github: https://github.com/google-research/google-research/tree/master/STraTA
  - on huggingface: https://github.com/huggingface/transformers/tree/main/examples/research_projects/self-training-text-classification
  - similar to other self training techniques

<img width="808" alt="image" src="https://user-images.githubusercontent.com/8890262/163197076-e64e19f1-a5e7-410e-bef0-e9cb45a145ba.png">

<img width="399" alt="image" src="https://user-images.githubusercontent.com/8890262/163197129-437cb2af-c632-4bed-9f6b-8f132489f58d.png">

## What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?
https://arxiv.org/abs/2204.05832

- best zero-shot performance, decoder only transformer trained on causal language modelling (think GPT-x)
- encoder-decoder model w/ mlm pretraining + multitask finetuning performed the best
- Compromise: decoder only model + causal lm + additional mlm (non causal lm)
- Question: does this generalize to IR? doesn't seem to be an evaluated task so not too sure
  - seems from recent papers (SGPT-BE, GTR) encoder only still performs better in fine-tuning, but SGPT generalizes better in BEIR

## Weak supervision for NLP (skweak, snorkle)
- define a set of labelling functions
- aggregate to create labels
- train model on weakly supervised labels
- [skweak](https://github.com/NorskRegnesentral/skweak)
- [snorkle](https://github.com/snorkel-team/snorkel)


## Synthetic Data Generation to Improve Model Performance
https://hackmd.io/gmDAH0fqRAKcZl3sPLdjsg

## Running inference with LLMs without enough cpu/gpu RAM
- https://twitter.com/huggingface/status/1524783489593360385
- huggignface accelerate for `load_checkpoint_and_dispatch`
- example google colab notebook: https://colab.research.google.com/drive/14wnxMvD9zsiBQo2FtTpxn6w2cpXCcb-7#scrollTo=y8Ne7jJdaF9F&uniqifier=1

### Using accelerate
- https://huggingface.co/docs/accelerate/big_modeling
- https://twitter.com/moyix/status/1546961979566325760?t=cMI_Yw37XzbncqzILBCgrw&s=09&fbclid=IwAR1E4eGinYDADIU9BdRtteWJHYBFqDpq6B0J18tgAB-2aP06FOYP0vO8IDw
- colab notebook: https://colab.research.google.com/drive/14wnxMvD9zsiBQo2FtTpxn6w2cpXCcb-7#scrollTo=SGS9OW5qnaUH&uniqifier=1

### transformers + accelerate + offload
https://twitter.com/moyix/status/1546961979566325760?t=cGtMefQd2_6_bKBW1KaJVQ&s=09&fbclid=IwAR1zQkZ-GF5SFTU1ev14kY6qEvXmLzHbuDNyxnlIV3qsP628FpBdx8WQudo

### transformers + accelerate + bnb int8
https://docs.google.com/document/d/1JxSo4lQgMDBdnd19VBEoaG-mMfQupQ3XvOrgmRAVtpU/edit?usp=sharing

## Training LMs with RL + human feedback
https://github.com/lvwerra/trl

## Inference using encoder-decoder/decoder only models for classification type tasks
- want to prevent hallucination
  - eg labels are `["positive", "negative"]`, but LMs can generate anything
  - want to only contrain output to labels at inference
- use constrained beam search
- https://huggingface.co/blog/constrained-beam-search
- Alternatively, use sum of log probabilities (see SGPT cross encoder)

## Training recipe for LLMs (100B model)
https://medium.com/yandex/yandex-publishes-yalm-100b-its-the-largest-gpt-like-neural-network-in-open-source-d1df53d0e9a6
- useful tricks for speeding up and stablizing training of large language models

## On debugging large scale training;
https://twitter.com/LoubnaBenAllal1/status/1542521845483655171?t=sfRLq876stw0CIKT0U0bKw&s=09&fbclid=IwAR2S5MAS8CFvTX3d4OeB_lrp-flH0gW_CbGMaMh-kG6JNe0-MQciTRA-4zM

## Prompt engineering for LLMs
- https://txt.cohere.ai/how-to-train-your-pet-llm-prompt-engineering/?utm_content=213394473&utm_medium=social&utm_source=linkedin&hss_channel=lcp-24024765
- https://docs.cohere.ai/prompt-engineering-wiki/?fbclid=IwAR0J08sW24dcBq_YpGpF4G0Pb-5OexCdlcscSsTlk28l5Ij2sXeQlg7GLrQ
- NER specific: https://towardsdatascience.com/advanced-ner-with-gpt-3-and-gpt-j-ce43dc6cdb9c#4010-fa6647c13fbe-reply

### using weird delimiters in prompts
https://twitter.com/goodside/status/1566899368535408640?fbclid=IwAR13iUzCVHnKpb2hX5kXJ2dsOvevspQATdPFnABHK0tDaGGQLgAxJ4kJUuI

### Resources for prompt engineering 
https://twitter.com/maria_antoniak/status/1569006194261659648?t=cRQDXuF3njbd040ZG8XTtA&s=09&fbclid=IwAR3tW69m60AzFdGZbqd5g0ZVKH2BVPyYPmb8EsN7PbScrS0LzKDkrHgjdQk

### AMA prompting for LLMs
- https://arxiv.org/abs/2210.02441
- Enables GPT-J-6B model to perform better than GPT3 on 15/20 benchmarks
- Github: https://github.com/HazyResearch/ama_prompting

### Self ask prompting
- paper https://ofir.io/self-ask.pdf
- code https://github.com/ofirpress/self-ask/blob/main/self-ask_plus_search-engine_demo.ipynb

## GLM-130B
- blog post: http://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/
- arxiv: https://arxiv.org/abs/2210.02414
- github: https://github.com/THUDM/GLM-130B
  - includes training + architecture recipe
  - Interesting finding: using embedding normalization (used in BLOOM) stabilizes training, but reduces downstream performance significantly
  - https://github.com/THUDM/GLM-130B/blob/main/logs/main-log-en.md
- leverage a unique scaling property of GLM-130B to reach INT4 quantization, without quantization aware training and with almost no performance loss, making it the first among 100B-scale models
- Improvements over GPT/PALM/OPT/BLOOM etc.
  - ROPE as positional encoding was better than ALIBI
  - mask in-filling + prefix lm objective
  - Deepnorm instead of pre-LN for layer norm improved stability in training
  - GeGLU instead of FFN (GLU + GELU activation)
  - No embedding norm (from BLOOM) since it lowered performance
  - Embedding Layer Gradient Shrink: identifies that the gradient norm can serve as an informative indicator of training collapses

## Improvements to train large language models
https://twitter.com/nlpguy_/status/1556881385927098369
- 2D positional encodings (GLM > GPT3)
  - from GLM experiments ROPE > ALIBI (may be due to infilling objective)
- allowing arbitrary order to predict spans
- Normformer
- P-tuning v2
- MoE (mixture of experts)
- Scaling data is as important as scaling model size (Chinchilla)
- In-filling maybe better than left to right decoding (GLM)
- GeGLU instead of FFN
- DeepNorm instead of pre-LN for layer norm
- don't use embedding norm




## Finetuning GPT-NeoX
https://nn.labml.ai/neox/index.html

## Finetuning models with only 20 examples (T-few)
- https://twitter.com/colinraffel/status/1560672781330567174?t=HOpz3AgPBtJJa4HAdm8dSA&s=09&fbclid=IwAR0pxKq39IcNfnZgLCTOJuGdW7U7ZVBOF-RlaUVBdMqNwj5jEAPiMB-ZZy0
- Few-shot parameter efficient finetuning
- paper: [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)
- github: https://github.com/r-three/t-few

## Efficient Methods for Natural Language Processing: A Survey
https://arxiv.org/abs/2209.00099

## Selective Annotation Makes Language Models Better Few-Shot Learners
- https://arxiv.org/abs/2209.01975
- Compared to state-of-the-art supervised finetuning approaches, it yields similar performance with 10-100x less annotation cost across 10 task
- How does this compare to t-few?
- Github: https://github.com/HKUNLP/icl-selective-annotation

## Alexa Teacher Models
- paper: https://arxiv.org/abs/2208.01448
- 20B seq2seq model (like T5)
- Better than large decoder only models on zero-shot superglue (except PALM)

## Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate
https://huggingface.co/blog/bloom-inference-pytorch-scripts?fbclid=IwAR32GkgjNi9Towdm6DwqzpWdotP5hXNv-WQl7ilM83xQ4ZmO9a48kQtriWI
- only provides deepspeed inference script

## Accelerate GPT-J inference with DeepSpeed-Inference on GPUs
https://www.philschmid.de/gptj-deepspeed-inference#3-optimize-gpt-j-for-gpu-using-deepspeeds-inferenceengine

## How to train a Language Model with Megatron-LM
https://huggingface.co/blog/megatron-training


## Building a Search-Based Discord Bot with Language Models
https://txt.cohere.ai/building-a-search-based-discord-bot-with-language-models/?fbclid=IwAR1-PzwBsNYI8IYqGl2mmxYolNGJUu_CEj75XTWklbyU0_-Zhbe78s9uDlc

## Efficient Few-Shot Learning Without Prompts
- paper: https://arxiv.org/abs/2209.11055
- github: https://github.com/huggingface/setfit
- pretrained models: https://huggingface.co/setfit
- finetune a pretrained sentence transformer on a small set of sentence pairs (can be as low as 2 example sentences per class) using a contrastive loss
- using ft sentence transformer to generate embeddings
- training a classifier head on top

## Training transformer language models with reinforcement learning (trlx)
- https://twitter.com/carperai/status/1577015392773414914?t=0Iqc4B0znD48FNFjqdAiTw&s=09&fbclid=IwAR0-tniAgUA_CHCJ1ryZyDZepRyH1SPX9X4FlTuDfRRAb4G8lx_IZpc8gFk
- https://github.com/CarperAI/trlx

## MoE + Weight sharing (WideNet)
- https://arxiv.org/abs/2107.11817
- Free speedup for training, more parameter efficient than MoE
- WideNet on HomebrewNLP-Jax: https://github.com/HomebrewNLP/HomebrewNLP-Jax/pull/85

## Considerations for code generation models (discussions from BigCode)
1. Architecture choice: AE, DE, Seq2seq
  - DE may work better for zero-shot, seq2seq better for finetuning
2. Training objective
  - AR (left to right) generation vs fill-in-the-middle (FIM)/causal masking
  - FIM seems more promising than causal masking?
3. Positional encoding
  - Rotary embedding vs ALIBI
  - ALIBI + FIM/causal masking may hurt performance since the tokens are rotated
    - may require "rotating" the matrix

Considerations:
1. FIM + rotary 
2. FIM + alibi
  - rotated vs un-rotated alibi
3. rotary + no FIM

Useful papers:
- [InCoder (FAIR)](https://arxiv.org/abs/2204.05999)
- [FIM (OpenAI)](https://arxiv.org/abs/2207.14255)


## Massive Text Embedding Benchmark (MTEB) Leaderboard
https://huggingface.co/spaces/mteb/leaderboard

## UL2
- paper: https://arxiv.org/abs/2205.05131v1
- uses a mixture of denoiser training objective
 - x-denoiser: extreme denoising
 - r-denoiser: short spans and low corruption
 - s-denoiser: sequential denoising/prefix lm objective
 - helps add data (better scaling laws?)
- enc/dec > decoder only
- weights https://huggingface.co/google/ul2

## Transcending Scaling Laws with 0.1% Extra Compute
- paper: https://arxiv.org/abs/2210.11399
- continues training a LLM (PALM) on a. few more steps with UL2's mixture-of-denoiser objective
- improves scaling properties

## Scaling Instruction-Finetuned Language Models
- instruction finetuning (not using RL from human feedback, but similar to T0's multitask finetuning)
- explores
  - scaling number of tasks
  - scaling model size
  - finetuning on chain-of-thought-data
- Flan T5 huggingface checkpoint: https://huggingface.co/google/flan-t5-xxl?text=Answer+the+following+yes%2Fno+question.+Can+you+write+a+whole+Haiku+in+a+single+tweet%3F
  
## InstructGPT
- instruction finetuning using RLHF 
- https://openai.com/blog/instruction-following/
- outperforms supervised finetuning
