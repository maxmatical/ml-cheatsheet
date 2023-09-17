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

## qs for Jonathan
1. how to evaluate models that finds out if overfitting to common benchmarks
    - benchmarks are sketchy
    - may require domain specific evaluations
2. masking loss in SFT
3. starting small and scaling up - any processes on how the training recipe should adapt as you scale up? data mixture, lr etc.
    - data choices
    - MuP?
