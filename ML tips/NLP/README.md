
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
  
[Natural Language Processing with Disaster Tweets](https://chrwittm.github.io/posts/2023-01-17-nlp-with-disaster-tweets/)
- smaller batch sizes helps model train more quickly
- **train on ALL data (train + dev) after hyperparam tuning/optimization**
  
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
    - another usecase (not in this article could be self-consistency filtering)
      - 1. train on all pseudolabeled data w/ hard labels
      - 2. predict on itself, keep only data where `pred == label`
      - 3. finetune on (can try different configs)
        - filtered pseudolabel data + labeled data, OR
        - filtered pseudolabel data -> labeled data OR
        - filtered pseudolabel data + labeled data -> labeled data (3 step finetuning)

### More pseudolabeling tips from same competition (1st place winner)
https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536?ref=mlcontests

Pseudo labels
Another major part of our solution is pseudo labeling. We applied 3 stages of pseudo labeling on the extra data from the previous Feedback competition. It was done in a leak-free manner for the individual folds and additionally for our models trained on all the data (6 versions of pseudo labels in total). The process consisted of the following steps:

1. Train an ensemble of models only on the given train data
2. Run predictions on the previous Feedback competition data using our full 2-stage pipeline
3. Use soft pseudo labels from this extra dataset and apply it to modeling in two different ways:
  - Concatenate pseudo labels with the actual labels in the given train data, and train simultaneously on all this data
  - Pre-train models on the pseudo labels and finetune it only on the given train data afterwards. Similar to: https://arxiv.org/abs/1904.04445
4. Repeat steps 1-3 three times using an ensemble of models trained on pseudo labels now

Apart from using previous Feedback competition data for pseudo labels, it was also used in some models as a pre-training dataset. The model was warmed up on the old data predicting the type of the chunk and further finetuned on the given train data.

### additional techniques
1. setting dropout to 0
    - all layers in transformers
    - may vary across arch/tasks
        - may need to experiment (may be case by case basis)
2. SWA
    - average weights across checkpoints
    - took task 3 checkpoints instead of a % of training epochs
    
### [Natural Language Processing with Disaster Tweets](https://chrwittm.github.io/posts/2023-01-17-nlp-with-disaster-tweets/)
- smaller batch sizes helps model train more quickly
- **train on ALL data (train + dev) after hyperparam tuning/optimization**


## Better pretrained models
https://ibm.github.io/model-recycling/

example: instead of using `roberta-base`, use `janeel/muppet-roberta-base-finetuned-squad` instead

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

## Prompting for LLMs
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

### Prompt design from Cohere (instruction vs examples)
https://txt.cohere.ai/generative-ai-part-1/

### providing diverse examples in prompts
https://twitter.com/xiye_nlp/status/1603821850592628738

### Openai cookbook
https://github.com/openai/openai-cookbook

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

## Training transformer language models with reinforcement learning (tlr/trlx)
- https://twitter.com/carperai/status/1577015392773414914?t=0Iqc4B0znD48FNFjqdAiTw&s=09&fbclid=IwAR0-tniAgUA_CHCJ1ryZyDZepRyH1SPX9X4FlTuDfRRAb4G8lx_IZpc8gFk
- https://github.com/CarperAI/trlx
- https://github.com/lvwerra/trl (PPO only)

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
- https://arxiv.org/abs/2210.11416
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
- Discussion on RLHF vs supervised finetuning: https://twitter.com/haoliuhl/status/1598774263166554112

## Constrastive search in transformers
- https://huggingface.co/blog/introducing-csearch

## Flash attention/xformers
- https://github.com/HazyResearch/flash-attention
- metaseq + flash-attention (megatron-lm + model parallelism) https://github.com/HazyResearch/
- Usage in other ml frameworks: https://github.com/HazyResearch/flash-attention/blob/main/usage.md
- xformers (similar): https://github.com/facebookresearch/xformers

## GPT-JT-6B (strong few shot classification performance)
- GPT-J + UL2 training objective + instruction finetuning (similar to Flan-t5)
- strong performance in few-shot prompting (HELM/RAFT)
- https://huggingface.co/togethercomputer/GPT-JT-6B-v1
  - see `Hosted Inference API` for example prompt
 
## Pretraining and finetuning large language models
https://hackmd.io/Cx0GK41RT5yPqNeeFwcT_g

key concepts covered:
- UL2
- FLAN-T5/FLAN-PaLM
- Instruction finetuning
- RLHF
  - trlx
- ILQL

## AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning
- Paper: https://arxiv.org/abs/2205.12410
- Code: https://github.com/microsoft/AdaMix
-  By only tuning 0.1-0.2% of PLM parameters, we show that AdaMix outperforms SOTA parameter-efficient fine-tuning and full model fine-tuning for both NLU and NLG tasks. 

## PubMedGPT
Idea
- 2.7B LM -> new SOTA on medical data
- standard transformer model on large amounts of text 300B tokens from the pile + full finetuning
- beats generic 100B+ models
- custom <10B parameter models trained on domain-specific data beats large generic models
  - **actually may not longer be the case, since beaten by med-palm**
  - 
twitter discussion
- https://twitter.com/percyliang/status/1603469265583353856
- https://twitter.com/MosaicML/status/1603436855067910147

blog post: 
- https://www.mosaicml.com/blog/introducing-pubmed-gpt
- https://crfm.stanford.edu/2022/12/15/pubmedgpt.html

code: https://github.com/stanford-crfm/pubmedgpt

huggingface model: https://huggingface.co/stanford-crfm/pubmedgpt


## SetFit
- few-shot learning using sentence transformers + contrastive loss + finetune classifier layer
- only needs 8 examples per class for really strong results
- github: https://github.com/huggingface/setfit
- compressing setfit models with kd and quantization: https://github.com/huggingface/workshops/blob/main/fewshot-learning-in-production/setfit-optimisation.ipynb

## self-instruct
tweet: https://twitter.com/yizhongwyz/status/1605382356054859777
arxiv: https://arxiv.org/abs/2212.10560
github: https://github.com/yizhongw/self-instruct

- bootstrapping PLMs to generate instruction, input, and output samples from a LM
- prunes samples
- use to finetune the original lm
- gets close to instructGPT-001
- also outperforms public instruction datasets by large margin (5%)
- almost annotation free method for aligning PLMs twith instructioins
- dataset included

## Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor
tweet: https://twitter.com/arankomatsuzaki/status/1605018849606533121
arxiv: https://arxiv.org/abs/2212.09689

- dataset generated from PLM for instructions
- not cleaned up, so contains noise
- despite containing noise, T5 trained outperforms open source datasets eg T0++ 
  - still underperforms FLAN-T5, but getting there
- unlike self-instruct, uses T5 models instead of GPT models


## How good of a BERT can one get in ONE DAY on ONE GPU?
tweet: https://twitter.com/giffmana/status/1608568387583737856
paper + code: https://paperswithcode.com/paper/cramming-training-a-language-model-on-a

what changes really mattered (gains on GLUE):
- 2% from arch changes (eg prenorm layer structure which also allowed higher LR)
- 1% from data
- 0.5% from training modifications

## Finetuine FLAN-T5 with deepspeed + huggingface
https://www.philschmid.de/fine-tune-flan-t5-deepspeed?fbclid=IwAR3b8PqZYpRHrJ_AVXJ-IIWCLAwBHYAUuKaADhbXKrwArUB50s1k0ceCZiM

## Parameter efficient finetuning (PEFT) methods for LLMs
- https://github.com/huggingface/peft
- Lora, P-tuning (v2), prompt tuning
- p-tuning v2 may be comparable to full finetuning
- how does it stack up vs (mixture of) adapters?

## Training GPT3 quality model for <$500k
https://www.mosaicml.com/blog/gpt-3-quality-for-500k

## Importance sampling of pretraining data to boost downstream performance
https://twitter.com/sangmichaelxie/status/1623397365443960832

## Instructor embedding model
- SOTA on MTEB 
- https://instructor-embedding.github.io/
- huggingface: https://huggingface.co/hkunlp/instructor-xl

## FLAN-T5 as a starting pt for few-shot prompting and finetuning
- currently SOTA of open sourced models
- https://twitter.com/tuvuumass/status/1620820330293100544

## TRL/TRLX + PEFT (lora) to train 20B param model with RLHF on 24gb gpu
- https://huggingface.co/blog/trl-peft?fbclid=IwAR1QqnI29DhMTMpuZ8OUFV_9HD9Z6_hCvu4cs-yOrfHf8hbGJr4i4lkglvc
- [can also do this in trlx with `delta_method`](https://wandb.ai/carperai/trlx/reports/trlx-LORA-support--VmlldzozMjgyMzcy)
  - more details see here: https://github.com/CarperAI/trlx/blob/main/trlx/data/configs.py#L52
  
## Finetuning Flan-T5-XXL (11b) on A10 gpu with PEFT + LoRA
https://www.philschmid.de/fine-tune-flan-t5-peft

## Alpaca - Instruction tuned LLaMa
- trained on 52k (unique) instructions generated via the `self instruct` method with GPT
- alpaca-30b available (finetuned on dataset with LoRA) https://twitter.com/aqaderb/status/1637828011130073088
  - hf model: https://huggingface.co/baseten/alpaca-30b
  - cleaned dataset: https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_cleaned.json
- alpaca-7b: 
  - adapters: https://huggingface.co/tloen/alpaca-lora-7b
  - base model: https://huggingface.co/decapoda-research/llama-7b-hf
 - code repo for training: https://github.com/tloen/alpaca-lora
  
  
## Transformer arch improvements in PALM
- https://twitter.com/rasbt/status/1637803700944093184
- multi-query attention (i.e. k, v weight matrices are shared
- parallelized transformer blocks
- swiglu activation
- ROPE embedding
- shared input/output embeddings. weight tying
- no biases in dense or layernorms for improved stability (esp important for layernorm?)
- sentencepiece tokenizer

## FLAN instruction templates:
https://twitter.com/abacaj/status/1633494842352214016

## Pretraining bert for $20
https://www.mosaicml.com/blog/mosaicbert

## Ultimate stack for PEFT for LLMs (focused on SFT here)
1. transformer model + load in 8bit (4bit in future?)
  - enable gradient checkpointing? or not needed with lora
2. [PEFT](https://github.com/huggingface/peft) library for `get_peft_model` and `prepare_model_for_int8_training`
  - example: [flan-t5-11b + peft single A10](https://www.philschmid.de/fine-tune-flan-t5-peft)
  - wrap the model and prepare for training
  - other examples provided in repo
3. Lightning or Accelerate as the trainer (lightning preferred)
  - lighting: can use either trainer or `L.Fabric` (which is equivalent to Accelerate), but can use colossal-ai
    - example of using Fabric for pretraining/finetuning llama: https://github.com/lightning-AI/lit-llama#finetune-the-model
    - lightning trainer also provides ability to use SWA callback, but can also just average last k checkpoints or something
    - Accelerate can't use colossalai, but can use deepspeed, may have issues with fsdp + cpu offload: https://github.com/huggingface/peft#peft---accelerate
  - pytorch lightning 2.0 also supports `torch.compile(model)` when `model` is a `LightningModule`
    - https://lightning.ai/pages/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/
4. If lora model still doesn't fit on 1 gpu (need cpu offload), or using multiple gpus with model parallelism, use colossal-ai (better than deepspeed)
  - see chatgpt replication using just colossalai: https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat
    - important bits: loramodule
  - also note: compatible with lightning: 
    - see docs: https://lightning.ai/docs/pytorch/stable/advanced/third_party/colossalai.html
      - few caveats:
        - need `configure_sharded_model` in lightning module (see example in colossalai repo below)
        - needs `colossalai.nn.optimizer.HybridAdam`
        - grad accumulation is only 1 for now
    - example using pytorch lightning: https://github.com/hpcaitech/ColossalAI-Pytorch-lightning/tree/main/benchmark/gpt
5. If RLHF on top of that, use [trlx](https://github.com/CarperAI/trlx)
  - [suports lora](https://wandb.ai/carperai/trlx/reports/trlx-LORA-support--VmlldzozMjgyMzcy) (but so does trl)
  - [supports 8bit adam](https://wandb.ai/jon-tow/trlx/reports/trlx-Add-bitsandbytes-optimizer-support-133--VmlldzozMjY1MzI1)
  - supports ilql method and nemo-megatron (ilql only for now)
  
## Finetuning LLM tips (#1 kaggle grandmaster) - focus on conversational data
- data: quality > quantity
- Foundation models: llama (non-commercial), `EleutherAI/pythia-12b-deduped` or `EleutherAI/gpt-neox-20b` are the best to use for finetuning for conversation right now.
  - recommend pythia over neox: https://twitter.com/BlancheMinerva/status/1649448167782449164
- LORA: use `bf16` for speed
  - for memory, use int8, gradient checkpointing, reduce batch size. should still get decent results
- **loss**: more consistent results only calculating loss on answer (i.e. masking out non-label input ids when calculating ce loss)
- prompt design: a hyperparameter to tune (using tokens, not using tokens, what kind of tokens etc.)

## A Theory on Adam Instability in Large-Scale Machine Learning 
https://arxiv.org/abs/2304.09871

mitigations for instability training llms:
- skipping batches. challenging to implement and require manual monitoring and intervention
- lower learning rate. but makes training longer (need more steps to reach same loss)
- lower `eps` in optimizer. could result in divide by 0 with low precision (fp16 etc.), so 1e-7/1e-8 might be good
- lowering batch size could help, but inefficient
- reduce `beta1` and `beta2` in adam optimizer (see mosaic), downside is could make update stale
- composition of data: higher quality data leads to less instability

## using deepspeed inference to get 2x inference speed on llms vs standard huggingface
https://twitter.com/abacaj/status/1649875255219847173

## Code data in LLMs seem to improve complex reasoning capabilities
https://twitter.com/abacaj/status/1647999551964323844

## Finetuning llms > prompting llms (should be fairly obvious)
https://twitter.com/rasbt/status/1646870571542536195

benefits of finetuning: https://twitter.com/abacaj/status/1646365774083244032
  - better model performance, examples become per of models internal knowledge
  - reduce cost of prediction (less tokens used)
  
## Recipe for training large models (LLM related)
twitter thread: https://twitter.com/borisdayma/status/1644358390313877504

report: https://wandb.ai/craiyon/report/reports/Recipe-Training-Large-Models--VmlldzozNjc4MzQz

mainly related to pretraining, not finetuning

key ideas:

General
- start with small model first
- don't stop experiments early, initially promising results could end up worse when fully trained
- experiment fast, keep notes/logs

Stability
- optmizer: shampoo for stability (but not in torch?), adamw with tips from https://arxiv.org/abs/2304.09871
- bf16 more stable than standard fp16
- weight decay might not be needed (or kept very low)
- no need for dropout
- normformer (stacking layernorms before and after attn + mlp) may help
- GLU may help
- gradient clipping by global norm to 1.0 or something

batch size and lr
- largest batchsize you can, use gradient accumulation or gradient checkpointing
- lr ports pretty well with model size increase in 5-10x increments
  - lr correlates with effective batch size: if multiply batch size by `k`, also multiply lr by `k`
- do a lr sweep of 3x/10x diff eg 1e-4, 3e-4, 1e-3
- optimize lr before expensive training run 
- use a lr warmup typically <=5-10%
- keep lr constant and from time to time increase/decrease lr by 3x increments
  - can use cosine/linear decay if lazy
  - cosine/linear decay to 0 near end of training
  - keep in mind mosaic ml just uses a standard linear decay after warmup so may not be need
  
logging
- Logging of training/validation loss + relevant metrics (accuracy…).
- Logging of parameters and gradients: 
  - The norm of parameters and gradients should regularly be logged so you can refer to it during instabilities
  - Histograms can be logged optionally (it will impact training speed but is nice to enable when debugging instabilities)
  - At start of training, manually verify that all gradients are flowing properly in your model and that parameters are being updated


example yamls from mosaic gpt:
- https://github.com/mosaicml/examples/blob/main/examples/llm/yamls/mosaic_gpt/30b.yaml
- https://github.com/mosaicml/examples/blob/main/examples/llm/yamls/mosaic_gpt/70b.yaml
```
# Optimization
scheduler:
  name: cosine_with_warmup
  t_warmup: 100ba
  alpha_f: 0.1

optimizer:
  name: decoupled_adamw
  lr: 8.0e-5
  betas:
  - 0.9
  - 0.95
  eps: 1.0e-08
  weight_decay: 0.0

algorithms:
  gradient_clipping:
    clipping_type: norm
    clipping_threshold: 1.0
```


## Vicuna
https://vicuna.lmsys.org/

## instruction flipping makes llms better zero-shot learners
https://arxiv.org/abs/2210.02969
- https://twitter.com/abacaj/status/1643416835843293186

## distillation > annoate + finetuning
- twitter thread: https://twitter.com/abacaj/status/1653977494196723712
- arxiv: https://arxiv.org/abs/2305.01645
- we find that distilling from T5-XXL (11B) to T5-Small (60M) leads to almost always a cost-efficient option compared to annotating more data to directly train a compact model (T5-Small (60M))
- **main idea**: given a fixed dataset + budget. better to spend the $ to train a large model on the data, and distill into smaller model, rather than spendig $ to annotate more data and finetune smaller model directly on combined data
- distillation done via KL divergence loss on **unlabeled data** (eg pseudolabeling)
  - q: would it perform even better if smaller model was finetuned on the labeled data afterwards? would be interesting to check

## MPT-7b
https://www.mosaicml.com/blog/mpt-7b

- 7b param model better than llama or at least competitive
- because trained on 1T tokens like llama
- Uses GPT-NeoX20B tokenizer (slightly better than standard gpt2 tokenizer)
  - set vocab size from 50,257 -> 50,432 (multiple of 128) and improved MFU by 4 percentage points
- Uses streaming dataset
- Uses ALIBI over positional encoding (improves stability)
- uses Lion optimizer over AdamW
  - more stable update magnitutes AND less optimizer state mem

##  Large-scale Near-deduplication Behind BigCode 
https://huggingface.co/blog/dedup

## Text Classification via Large Language Models
https://arxiv.org/abs/2305.08377

