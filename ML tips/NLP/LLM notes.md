# LLM training

maybe something like
1. continue pretraining on in domain data (domain adaptation)
    - starting from some 7b model
    - can we do 30-40b? like falcon, mpt etc (i wish)

2. IFT using openai
    - can use synthetic data generation
    - rag
    - etc.
    - maybe use some form of EVOL (wizard-lm)/self-instruct
        - new thing: instruction backstranslation?
    - dataset doesn't have to be large (as shown by other works)
        - only a few examples per instruction should be sufficient?
    - distilling from openai via KL divergence doesn't work because you can't get logits
    - structure in FLAN format
    - flipping labels/predicting instruction
        - https://twitter.com/Yampeleg/status/1679476706434752512
        - original: given customer question, what is the reply
        - flipped: given agent reply, generate a relevant customer question
        - predicting instruction: given input/output (from some non-agent reply data like intent, self-instruct etc) what was the instructiion

3. RLHF data collection
    - can collect
        - comp vs openai on IFT data
        - comp vs itself on IFT data
        - new data:
            - collection?
    - more synthetic data like evol/self-instruct
    - launch it to cx agents and let them play with it to get data
        - return 2 results and use user signals as data

4. RLHF training

    - maybe use https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
    - total available gpu/6 = sum of b in parameters for actor and critic model
        - eg 4 * a100-40 / 6 = 26b params total 


# consider using flash attention
https://github.com/OptimalScale/LMFlow/pull/536/files
`--use_flash_attention True` 
- needs to updated to that version of lmflow

# mpt-7b w/ gradient checkpointing
use this checkpoint
- https://huggingface.co/cekal/mpt-7b-peft-compatible

# llm leaderboard
https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

# inspo xfinance 
https://www.stochastic.ai/blog/xfinance-vs-bloomberg-gpt
500m tokens for domain adaptation then ift on like 80k examples

# disable_group_texts for SFT/IFT
- https://github.com/OptimalScale/LMFlow/pull/577

# maybe consider using this
- domain adaptive pretraining: LMFlow w/ block size
- finetuning with: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat 
    - only main consideration is how weights are saved w/ deepspeed chat since not using hf trainer
    - SFT maybe can still use LMFlow text2text but see https://github.com/OptimalScale/LMFlow/pull/577
- use deepspeed chat over something like trl/trlx because missing EMA and mixing unsup data (like instructGPT)
- rlhf training experiences: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training#training-experiences
    - find that mixing unsup data caused model divergence, so maybe not a good idea to mix in?

## alternative frameworks for training
https://twitter.com/abacaj/status/1692957942515974236

pretrain:
- axolotl: https://github.com/OpenAccess-AI-Collective/axolotl
    - has flash attn, seq packing, deepspeed
    - see the configs
- LMFlow:
    - similar to axolotl
- megatron (big science fork)
    - 3D parallelism
    - https://github.com/bigcode-project/Megatron-LM

SFT/IFT:
- axolotl
- LMFlow (maybe?)
- Deepspeed chat

RLHF:
- deepspeed chat
    - EMA

Q: if we mostly use deepspeed for training (i.e. no megatron) can we leverage EMA from deepspeed chat?

## choice of optimizer is axolotl (deepspeed)
- seems like 8bit adam doesn't play well with fsdp with offload
- memory seems neglibable with deepspeed
- probably just keep with deepspeed's adamw 

## sequence packing with axolotl (SFT)
```
# use efficient multi-packing with block diagonal attention and per sequence position_ids. Recommend set to 'true'
sample_packing: true
```
# domain adaptive pretraining data dedupe
- https://huggingface.co/blog/dedup?fbclid=IwAR0DLo7AJigg15-77fV2S89Cc3bFpYhWDPtJJwyFTW7JZMdjw0DuQ2sxv0o

- an embedding based approach: https://twitter.com/abacaj/status/1689293981668999169
    - generate embeddings
    - enumerate through text: get similarity matrix w/ something like faiss
    - remove anything where similarity >0.9-0.95
        - keep the longer answer?
        - longer answer better detailed/step by step solutions
    - then sort by ascending order
        - when training, no shuffling
        - think cirriculum learning
        - faster training due to pregressive seq size
        - some form of cirriculum learning? longer instructions more complex, so starting from simpler instructions first

# better pretraining loss could be a good indicator of downstream performance
- https://huggingface.co/papers/2308.01825 
- better LM loss -> better math reasoning
- could translate to our usecases

# llm patterns
https://eugeneyan.com/writing/llm-patterns/

# serving with vllm

# using ema in rlhf
- can we extend this to training as well?
- using `moving_average` https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/main.py#L485
- `moving_average` https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/utils.py#L223
- seems like EMA model is getting initialized via deepspeed
# instruction back translation 
https://twitter.com/jaseweston/status/1690888779878330368

# bredth first pipeline paralellism
https://arxiv.org/abs/2211.05953

# maybe megaton > deepspeed for pretraining?

## megatron deepspeed
- good overview of deepspeed vs megatron vs megatron-deepspeed https://twitter.com/StasBekman/status/1636089783905841152
- megatron-deepspeed: DP + TP + PP + Zero-1
- https://github.com/bigscience-workshop/Megatron-DeepSpeed
- also uses seq packing: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/utils.py#L253

## megatron LLM
- https://github.com/epfLLM/Megatron-LLM
- documentation
- Used by OpenAssistant
- supports finetuning of LLMs (llama, falcon, code-llama)
- 3d parallelism
- pretraining and SFT
- GQA and MQA
- ROPE, RMSNorm, LIMA-dropout
- FA2
- BF16
- Conversion to/from HF hub
- DOESN'T have sequence packing

## UL2 in megatron training
- https://twitter.com/YiTayML/status/1648631971457925122


# overfit to SFT/IFT data
- https://twitter.com/ArmenAgha/status/1692815116688068725
- eval loss don't correlate that well with actual use
```
Actually over fitting on your instruction tuning set seems to give the best human evaluation performance
```

# starcoder as a strong starting point
https://twitter.com/abacaj/status/1693680157570080782

# openMOE
https://xuefuzhao.notion.site/Aug-2023-OpenMoE-v0-2-Release-43808efc0f5845caa788f2db52021879
- some MOE insights
- insight into using UL2 as pretraining objective

# data generation, prep, and filter strategies
https://twitter.com/Shahules786/status/1693679663548182894
- data dedupe w/ similarity search
- gradient-based pruning
- wizard lm
- instruction backtranslation: https://arxiv.org/abs/2308.06259
- distilling step by step
- rag for code gen (code t5): https://arxiv.org/abs/2305.07922
- orca
- rest: https://arxiv.org/abs/2308.08998
    - train on model's own outputs 
    - a form of pseudolabeling
    - https://twitter.com/Yampeleg/status/1693744585539780998
- https://twitter.com/Yampeleg/status/1693729509072154963
- unnatural instructions: https://arxiv.org/pdf/2212.09689.pdf
    - used for unnatural code llama
    - probably used gpt4 for unnatural code llama
- SORT INSTRUCTIONS FROM SHORT -> LONG
    - 

# pretraining objective with UL2

# lr suggestions
- for finetuning only? or pretraining works too?
- 0.0003*(n_gpus * per_device_bs * max_seq_length * pct_unmasked_tokens/4_000_000)^0.5
    - seems to only be for sequence packing
    - upper-bound

# whether to mask input in IFT/SFT
- https://twitter.com/Yampeleg/status/1694156329261158609
    - suggest training on full text (instead of just output like alpaca) is better
    - Philipp Singer: opposite experience
- probably want to mask input on val set
- TLDR: worth experimenting both ways and have a good evaluation framework to determine which is better


# DPO - discussions w/ Chris Manning
- trl trainer
- definitely want SFT first
- no real large scale replication yet, Meta AI is exploring
- seems to be stable, no real "tips and tricks" like in instructgpt paper (yet)

# Manning take on RLHF
- from Schulman from openai
- SFT is limited when you want to prevent the model from doing something (eg hallucination), since it's only seeing positive examples it will always try ot generate the most likely tokens
- RLHF/DPO etc. explicitly show negative examples to show the model how NOT to behave

# sequence packing in SFT
- does it do better?
- may need to reduce the LR by 1/2
    - or increase lr since the effictive batch size is larger? (larget bs ~= lower lr)
    - the point is lr shouln't stay the same
- loss may end up higher simply because there's more tokens per batch

# D4 for LLM pretraining
- https://arxiv.org/pdf/2308.12284.pdf
- https://twitter.com/kushal_tirumala/status/1696632999134273927
- new data selection method 
- better than randomly sampling minhash deduped web docs

- 3 steps
1. embed data pts w/ small LM + cluster w/ Kmeans
2. remove closest neighbor of each data point
3. re-cluster and from remaining pts choose furthest from centroids

Data curation details
- start from CC
- apply MinHash (Spark impl of MinHashLSH) to dedupe
- -> CC-dedup 600m docs in total
- sample roughly 100m documents to calculate centroids
- embed each document by feeding it into a 125M OPT model and use the last-layer embedding of the last token
    - each vector is of size 768 and is NORMALIZED
- use 
```
faiss.Kmeans(
    768, # 125M OPT model embedding size
    11_000, # 11k clusters
    niter=20, # 20 iterations
    verbose=True,
    seed=0,
    gpu=False,
    spherical=True,
    min_points_per_centroid=1,
    max_points_per_centroid=100_000_000, # roughly eq to sample of 100M docs
)
```
- 11k clusters chosen due to heuristic that number of clustesr should be roughly be equal to sqrt of total number of points being clustered
- don't explicitly cluster balance
- apply semidedupe with r_dedupe=0.75: https://arxiv.org/pdf/2303.09540.pdf
    - We tune the deduplication threshold epsilon for each dataset manually to get the desired deduplicated
dataset size.
- after deduplication, we re-cluster the data in the pruned dataset
    - note: is this done with the 600M*0.75 dataset, or done with another 100m Sample?
    - https://arxiv.org/pdf/2206.14486.pdf
    - remove data by increasing order of distance to nearest cluster centroid until we keep r_proto % of remaining data
    - should be pretty simple: have all 600M data points and their dist l2 dist to the cluster centroid and remove top k%


# lr scheduler
- https://twitter.com/giffmana/status/1699113666170208511
- linear/cosine + fixed duration is best
- maybe constant/rsqrt to find best duration then use linear/cosine

## sgdr discussions
- https://twitter.com/jeremyphoward/status/1698853297124507929
- Jonathan Frankle (mosaic), Lucas Beyer (google brain), DM/Meta all find multiple cycles of cosine decay performs worse than single cycle
    - Eleuther running some tests on sgdr
    - sidenote: standard ensembles may perform better than snapshot ensembling


# Mu parameterization
- https://twitter.com/andrew_n_carr/status/1701361841577111980
- used for lr and weight initialization
- used by openai?
- github https://github.com/microsoft/mup
- may not be used in practice
    - def not in RLHF (from Louis)
- limitation: the LR is constant so not sure how it scales w/ lr schedulers

## cerebras training LLMs with muP
- https://docs.cerebras.net/en/latest/wsc/how_to_guides/mup_docs.html#transfering-hyperparameters-with-transfer
- follow the formulas in table 14 of the `Cerebras-GPT paper` from the proxy (small) model

## new cerebras cerebras/btlm-3b-8k-base
- new cerebras model trained w/ muP
- https://arxiv.org/abs/2309.11568
- https://huggingface.co/cerebras/btlm-3b-8k-base
    - `param_groups = model.get_mup_param_groups(lr=1e-3, weight_decay=0.1)`
    - https://huggingface.co/cerebras/btlm-3b-8k-base/blob/main/modeling_btlm.py#L550
    - fairly straightforward
- batch size is an important hyperparameter too! since max bs can be different
    - may need to keep batch size the same as the large (target) model

### to use
- must manually tune (for the proxy model)
    - lr
    - m_embed
    - std_base
    - note: this doesn't include `attention_temperature` or `output_temperature`
    - can also try linear vs cosine decay as well
- when scaling up:
    - follow the formulas for scaling up in https://arxiv.org/pdf/2304.03208.pdf - Table 14
    - see https://huggingface.co/cerebras/btlm-3b-8k-base/blob/main/modeling_btlm.py for `mup` code changes
        - `get_mup_param_groups` for learning rate related
        - see the code for initialization + output logits multiplier

## muP tips from one of the authors
https://twitter.com/ibab_ml/status/1705423643420119194
1. Keep the initialization and learning rate on your embeddings fixed as you increase the width of the model. Do the same for all biases in the model.
2. For all linear layers, multiply the learning rate with 1/width and the initialization scale with 1/sqrt(width).
3. Scale the attention logits with 1/head_dim instead of the usual 1/sqrt(head_dim). Tip: just leave the head_dim fixed and increase the number of heads. Then you can ignore this step.
4. Multiply the final output logits of the transformer with 1/width. This is equivalent to changing the temperature of the softmax.
5. (Optional) Initialize the final output layer with zeroes for slightly better results.

- results only hold for transformers trained w/ Adam


## muP in GPT-NEOX
https://github.com/EleutherAI/gpt-neox/blob/main/README-MUP.md

- see also the paper Appendix F.4
    - trained proxy on 4-16b tokens instead of 300b final tokens

- should follow example like this https://github.com/microsoft/mup/tree/main/examples/Transformer#%CE%BCp-transformer


- seems to be an issue w/ mup: https://github.com/EleutherAI/gpt-neox/issues/956

# phi 1.5 - Data is all you need
- https://arxiv.org/abs/2309.05463
- filters data using technique from `textbook is all you need` https://arxiv.org/abs/2306.11644
- filtering data into an "instructive" dataset
- GPT3.5 is used to generate synthetic content
- GPT4 is used to annotate a small subset of python data
    - prompt is `determine its educational value for a student whose goal is to learn basic coding concepts`
    - using embeddings from `codegen` train RandomForest classifier to classify and filter out data that has low educational value
- combines Phi1 dataset (7b tokens) with new synthetic "textbook-like" dataset (from gpt3.5 20b tokens) and filtered Falcon refied web dataset (88b tokens) in 20/40/40 split

Key takeaways
- using high quality data > more tokens (?)
    - RESULTS SHOULD BE TAKEN WITH A LARGE GRAIN OF SALT
- using gpt4 to filter, codegen as embedding
    - codegen is code specific
    - can combine with other methods like D4? or this paper from cohere: https://arxiv.org/abs/2309.04564
- synthetic data?

# optimizer choices for finetuning
- SGD may be preferable for finetuning over adam
    - less memory, less tunable parameters, not much difference in results
    - try using a much lower LR (like 10x lower?)
    - may not be suited for pretraining though!
    - empirically found not to give good results, but still worth testing in mem constrained scenarios

- maybe the above also applies to lion?

- some finetuning tips: https://twitter.com/giffmana/status/1634828535729618944
    - low lr
    - no wd
    - zero init new linear head

- momentum *may* be needed to get higher quality models, see: https://twitter.com/_arohan_/status/1630062139259101185

# Advice from Jonathan Frankle (LLMs)

## general
- start small - start w/ 125M model instead of 70b model, work your way up
- be skeptical about what you read
- be skeptical of your intuition
- test everything for yourself

## evaluation
- can't make any decisions until you know what success look like
- not a 1 size fits all process

on benchmarks like mmlu
- relying on benchmarks are sketchy
- may require domain specific evaluations

## data
- pre-training
- synthetic data: Jonathan skeptical
    - can be bootstrapped from IFT? eg llama 2 paper
    - require some level of human in the loop like filtering/curation

- deduplication:
    - removing highly repeated examples may be a good idea
    - removing semantically similar examples
    - jury is still out on whether it works
        - do the work and test on your own data!

- cirriculum learning/token mixing
    - nothing super concrete wrt cirriculum
    - constructing batches - open question atm
    - lower order concern? focus on data source first

## pretraining vs finetuning
- start w/ pretrained model + finetuning if required
- go to pretraining only when required

## hyperparam choices
- pos embedding, activations, optimizer, tokenizer choices
- pick what's proven to work (pick what's popular)
    - picking llama2 choices won't go wrong
- thoughts on sgd: 
    - sgd > adam in vision(?) if well tuned
    - hard to go wrong with adam
    - can try if lots of compute budget


## qs for Jonathan
1. how to evaluate models that finds out if overfitting to common benchmarks
    - benchmarks are sketchy
    - may require domain specific evaluations
2. masking loss in SFT
3. starting small and scaling up - any processes on how the training recipe should adapt as you scale up? data mixture, lr etc.
    - data choices
    - MuP?


# why we may not want to use prefixlm/UL2
https://twitter.com/haileysch__/status/1691483230761857024

- CLM better at inference

# LLM guides and tools by Stas Bekman (big science)
- https://github.com/stas00/ml-engineering/tree/master#machine-learning-engineering-guides-and-tools
- the log books may be particularly useful for LLM training


# gradient checkpointing w/ llama 2
- need to pass `use_reentrant=False` to `checkpoint()`

# distilling step by step from larger models
https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html?m=1

# torch distributed utils
https://github.com/crowsonkb/torch-dist-utils


# qwen 14b
- https://twitter.com/Yampeleg/status/1706967038000763164
- qwen technical report

---
Implementation

Several details caught my eye while reading it's paper.
These might be interesting for anyone involved with training models at this scale.

---
Tokenizer

Like many other multilingual LLMs, they had to add language specific tokens to their tokenizer.

- Used GPT-4's tokenizer (CL100k)
- Spllited numbers into single digits.
- Added Chinese common words.
- The final tokenizer has a vocab of 152K.
- Result: The compressive rate of this model is better than all the competing models.

---
Pretraining

Used a diverse set that covers a wide range of topics from various sources, including: Web documents, encyclopedia, books, codes, etc.

- The actual sources are not listed.
- They do mention that they extract text from HTML pages and use language identification tools to determine the language.
(Hinting: Mass scraping)

Deduplication methods:
- Exact-match.
- MinHash.
- LSH.

Filtering methods:
1. Rule based.
2. ML based:
- ML models train to estimate: Text-quality (scoring), Identify inappropriate content.

By hand:
- They manually sample texts from various sources, review them by hand to ensure high quality.
- They selectively up-sampled specific datasets coming from certain sources.

Hyperparameters:
- AdamW: betas = (0.9, 0.95), eps = 1e-8, cosine scheduler, min_lr = 10% of max_lr (Aggressive choice!), BF16.

---
Supervised Fine-Tuning

As said, they further fine-tune the base model for specific domains.

In the process, they use:

- Self-instruct for generating synthetic high quality data.
- For evaluation, they tested for code excitability (Running the generated code).
- There is also a usage example demonstrating the ability to "Auto Comment" a code snippet, on this topic:

In my own opinion, this is one of the "secrets" of ChatGPT: Generating comments BEFORE each line of code and use the commented code as a training data.

You can easily see that all the code ChatGPT generates follow this structure and it is obvious why this type of format boosts coding performance. 

---
Architecture Tweaks

RoPE:
Like everyone, they use RoPE (One of the best options today) but they are using FP32 precision for the inverse frequency matrix, rather than BF16 or FP16.

> Noted.

[1] RoPE is (VERY) powerful: https://blog.eleuther.ai/rotary-embeddings/

Bias:
Again like everyone, they removed all biases [1] nearly everywhere BUT they added biases for the QKV layers [2].

[1] Bias removal improves performance: https://arxiv.org/abs/2204.02311
[2] RoPE + Bias extrapolation study: https://spaces.ac.cn/archives/9577

Normalization:
RMSNorm. Pre-Norm.

Nothing new under the sun but they do hint that better methods exist today other than pre-norm and will be experimented with in the future. They might be hinting at this: https://github.com/ZixuanJiang/pre-rmsnorm-transformer (?)

---
Context window length

- Using dynamic NTK-aware interpolation during inference. [1]
- Specifically, they used YARN's version: dynamic interpolation version that changes the scale by chunks. [2] 
- Interesting observation: Long-context modeling ability varies across layers - Lower layers being more sensitive in context length extension compared to the higher layers. To leverage this: They shorter windows for lower layers and longer windows for higher layers.

[1] Dynamic NTK: https://reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
[2] YARN: https://arxiv.org/abs/2306.15595

---
Synthetic Data Generation

Self-instruct: They used the few-shot capabilities of the model to generate data [1], But they do this in an iterative way: At each generation step, they manually remove noisy samples and retrain the model.

Doing this process multiple times resulted in a much more reliable model for fewshot generation. [2]

[1] - Giving it couple of examples and let it "complete" the text to generate more.
[2] - This is probably because of the main issue with synthetic data: Filtering. If you do not filter the data well enough you amplify the noise and not just the signal.

## qwen vl - vision langauge model
- https://arxiv.org/abs/2308.12966
- maybe a good insight into multi-modality?
- 

# Mistral AI 7b
- blog: https://mistral.ai/news/announcing-mistral-7b/
- github: https://github.com/mistralai/mistral-src/blob/main/mistral/model.py
- huggingface: https://huggingface.co/mistralai/Mistral-7B-v0.1

Summary:
- beats llama2 13b 
    - maybe qwen 14b as well?
- uses grouped-query attention (may be better than MQA)
- uses sliding window attention (SWA)
    - each layer attends previous `4096` hidden states
    - A token `i` at layer `k` attends to tokens `[i-sliding_window, i]`
    - SWA + flash attn = 2x speed improvement on seq len of 16k with window of 4k
    - see their `cache.py` and `model.py` for implementation
- uses `Byte-fallback BPE tokenizer`

- reference attention implementation: https://github.com/mistralai/mistral-src/blob/main/mistral/model.py


# Sliding window (local) attention

## reasons why maybe local attention isn't worth the hype
- according to Jonathan (mosaicml) attention cost becomes more efficient as you scale up model size
- eg going from 4k -> 8k at 30b is only 15% increase in cost
- so maybe only super important at 7b scale?

- from meta paper: https://ai.meta.com/research/publications/effective-long-context-scaling-of-foundation-models/
    - pretraining at 4k context, then extending to 32k is better than training from 32k
    - uses yarn
    - so maybe local attention isn't that useful?


## local attention implementations
- reference attention implementation: https://github.com/mistralai/mistral-src/blob/main/mistral/model.py


- mistral huggingface model: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
    - just use the prepare mask fn at the main model level
    - can maybe even do something similar in megatron? just the main model `forward`


# AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model
- https://arxiv.org/abs/2309.16058
- Given some modality encoder (vit etc.) shape is `bs, embed_dim_encoder`
- use a projection layer (adapter) to project to shape shape `bs, n_tokens, embed_dim_llm` where `n_tokens` is between 64-256, set `n_token` per modality adapter
    - different ways of using adapters, see flamingo/IDEFICs, LLaVA, QWEN-VL, MiniGPT-4 etc.
    - maybe some positional encoding (1d or 2d) are used/needed
- depending on training recipe, certain parts of the joint model (llm, encoder, adapter) are frozen during training

# pile dedupe
https://github.com/EleutherAI/pile_dedupe


# filtering chat datasets
https://twitter.com/Yampeleg/status/1708080098719502724

Let's say that we want to use StackExchange as multi-turn chat dataset. [0]

Problem:
Wild internet comments can be uninformative.

Solution:
Use a "backward model" [1] and take it's log_probs as relevancy indicator.

Intuition:
If a source message can not be predicted from it's response, this response could have been responded to many other sources.

Example of Uninformative Response:
"Yes"

---
Trick Name: Mutual Information Maximization.
Read more: https://arxiv.org/pdf/1911.00536.pdf

-
[0] Many famous Huggingface datasets include stack exchange but this can also be applied to any chat dataset.
[1] Backward: Predicts the source message from it's response.

# slim pajama dataset
https://twitter.com/Yampeleg/status/1707833483119231127

- using MinHashLSH
1. Remove all punctuation, consecutive spaces, newlines, tabs, and leading or trailing escape characters.
2. Convert everything to lowercases.
3. Tokenize to 13-grams.
4. Filter by Jaccard similarity with threshold of 0.8 filtering.

# LOOK AT THE DATA!
- manually inspect the data 
    - gain further insight into the data
    - to remove bad quality samples
- can be maybe combined with training smaller model?
    - eg using a model to find the low quality samples
- look at the bad samples
    - highest loss/perplexity samples
    - examples where model predicts wrong in classification setting 
        - (error analysis)
    - etc. 
- look at the data being removed/duplicates

## possible pretraining scenario for a data engine
1. use (small) model train on all data
2. use some metric (like perplexity) to identify bad sample candidates 
3. use human/ai to judge whether to throw out the sample
    - alt: just throw out if low enough perplexity
4. train (larger) model on the filtered data
5. repeat 2-4?

## possible scenario for SFT data engine
1. have a set of prompts for SFT without corresponding outputs (the "unlabeled" data)
2. generate for input w/ llm (think pseudolabels for classifier)
3. manually inspect + correct output
    - alt: use gpt4 as a judge to identify and throw out (or manually relabel) bad outputs
    - alt: use some metric of confidence (perplexity?) to identify the low confidence samples to re-label or remove
4. add back into dataset

Q: why not just distill from a more powerful model like GPT4?
- can generate input and output (assuming high quality)
A: maybe due to legality concerns. or assume domain specific model > GPT4 like model that's more general

## RLHF
- RLHF/RLAIF by itself is kind of a data engine, since you're training the RM on actual human/ai annotations, and model is used to generate samples to further enhance RM (like LLAMA rlhf)

# Process supervision to improve LLM reasoning
- RLAIF for Process Reward Model (PRM)
- https://arxiv.org/abs/2305.20050 (openai)

# Ensemble of RM may help in RLHF/RLAIF
https://arxiv.org/abs/2310.02743
- ensembles for pseudolabels work! 
- how can we take advantage of this?

# LLM data engine
- idea: model -> annotate -> data -> train -> model
- really just for SFT?

## possible scenario for SFT
1. have a set of prompts for SFT without corresponding outputs (the "unlabeled" data)
2. generate for input w/ llm (think pseudolabels for classifier)
3. manually inspect + correct output
    - alt: use gpt4 as a judge to identify and throw out (or manually relabel) bad outputs
    - alt: use some metric of confidence (perplexity?) to identify the low confidence samples to re-label or remove
4. add back into dataset

Q: why not just distill from a more powerful model like GPT4?
- can generate input and output (assuming high quality)
A: maybe due to legality concerns. or assume domain specific model > GPT4 like model that's more general

## possible pretraining scenario for a data engine
1. use (small) model train on all data
2. use some metric (like perplexity) to identify bad sample candidates 
3. use human/ai to judge whether to throw out the sample
    - alt: just throw out if low enough perplexity
4. train (larger) model on the filtered data
5. repeat 2-4?

