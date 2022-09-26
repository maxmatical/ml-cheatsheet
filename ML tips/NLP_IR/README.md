# IR Related stuff


## IR models on zero shot BEIR, ranked by performance
https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit#gid=0



## Information Retreval/Neural Search/Open Domain QA

Cross Encoder for stronger perfomance, Bi-encoder methods faster

## Notes for training and using bi-encoders
 - if possible, use the same model for query and document encoders (same arch, weights? etc.). sometimes it may not be possible eg in multi-modal cases
 - mean pooling seems to work better than other pooling strategies
 - a hybrid approach (combining BM25 w/ dense model) could work really well


## Finetuning Bi-encoders
framework: [haystack](https://github.com/deepset-ai/haystack)

haystack dpr notebook: https://github.com/deepset-ai/haystack/blob/master/tutorials/Tutorial9_DPR_training.ipynb

framework: [Margin MSE distillation](https://github.com/sebastian-hofstaetter/neural-ranking-kd)

pretrained models: https://huggingface.co/models?other=dpr&sort=downloads

framework: [matchmaker](https://github.com/sebastian-hofstaetter/matchmaker)
  -  for cross-arch kd

## useful pretrained models

Sentence Transformer Models:
- https://www.sbert.net/docs/pretrained_models.html#semantic-search
- Multi-qa models seems to be quite strong

coCondenser (current SOTA on NQ and MS-MARCO): https://arxiv.org/pdf/2108.05540.pdf
- github: https://github.com/luyug/Condenser
  - Fine-tuning cocondenser for DPR/ODQA (NQ, TriviaQA), use haystack/fb dpr repo + cocondenser weights
  - Fine-tuning cocondenser models on MS-MARCO: https://github.com/texttron/tevatron/tree/main/examples/coCondenser-marco
- HF pretrained models: https://huggingface.co/Luyu
  - note these are only pre-trained models (so haven't been fine-tuned for ODQA/IR)
  - `Luyu/co-condenser-marco-retriever` is a fine-tuned retriever 




## SOTA literature to keep track of in IR/ODQA
Can use BEIR dataset as a guide: https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit#gid=0

1. coCondenser
- https://arxiv.org/pdf/2108.05540.pdf
- MSMarco Results
 - MRR@10: 38.2
 - R@1000: 98.4
- NQ:
 - R@5: 75.8
 - R@20: 84.3
- github: https://github.com/luyug/Condenser
  - Fine-tuning cocondenser for DPR/ODQA (NQ, TriviaQA), use haystack/fb dpr repo + cocondenser weights
  - Fine-tuning cocondenser models on MS-MARCO: https://github.com/texttron/tevatron/tree/main/examples/coCondenser-marco
- HF pretrained models: https://huggingface.co/Luyu
  - note these are only pre-trained models (so haven't been fine-tuned for ODQA/IR)
  - `Luyu/co-condenser-marco-retriever` is a fine-tuned retriever 
- Easy to plug and play using Haystack/dpr training repo 
- may not work as well in adaptive pre-training? performs worse than base model/mlm

2. YONO
- https://arxiv.org/abs/2112.07381
- github: currently none
- competitive/slightly better performance on ODQA (NQ/TriviaQA) using **just** retriever
- no results on MS-MARCO
- better performance when adding Ranker as well

3. Colbert v2
- https://arxiv.org/abs/2112.01488
- github https://github.com/stanford-futuredata/ColBERT/tree/new_api
- slightly better than coCondenser on MS-MARCO, no results in NQ/TriviaQA

4. Sentence transformer Models:
- https://www.sbert.net/docs/pretrained_models.html
- `multi-qa` and `msmarco` models for IR
- possilbly `all-*` models like `all-mpnet-base-v1` and `all-roberta-large-v1`

5. Combining dense embeddings with sparse models (BM25/Splade v2 etc)
- combining sparse and dense models: 2 runs
  - 1 run with sparse retrieval (eg BM25) and 1 run with dense (dpr/cocondenser etc.)
  - linear combination of the 2 scores as the new score (similar to score combiner work)

6. LaPraDoR
- https://arxiv.org/abs/2203.06169
- unsupervised
- BEIR Results:
 - .438 nDCG@10
- claim: doesn't need to fine-tune on a downstream dataset
- also uses mean pooling (like GTR, Sentence T5)

7. Spider
- https://arxiv.org/pdf/2112.07708.pdf
- zero-shot performance better than cocondenser
- similar in-domain (finetuned) performance as cocondenser

8. InPars:
- https://arxiv.org/abs/2202.05144
- Use GPT3 to generate questions from a document (simlar to GPL)
- Fine-tune on synethetic + real data
- also see GPL

9. QGen
- https://aclanthology.org/2021.eacl-main.92/
- Similar idea to InPars, but uses a finetuned encoder-decoder model instaed of zero-shot
- see GPL for domain adaptation on top of QGen

10. GPL:
- https://arxiv.org/abs/2112.07577
- Take a T5 finetuned on MS MARCO to generate queries, apply to downstream datasets to generate questions from docs for synthetic training samples
- BEIR results
 - base: 0.445 NDCG@10
 - Upper bound (TAS-B + GPL): 0.459 NDCG@10

11. GTR (Generalizable T5-based dense Retrievers)
- https://arxiv.org/abs/2112.07899
- BEIR Results:
 - between **0.416 - 0.458 on BEIR NDCG@10** 
- MS Marco Results:
 - **nDCG@10 0.442** (currently SOTA)
- Scaling up encoder, but keeping bottle neck embedding size fixed at 768
- Scaling T5 (using only the encoder), use multi-stage training
 - multi-stage training and architecture taken from [Sentence-T5](https://arxiv.org/abs/2108.08877)
 - mean pooling seemed to work the best (The sentence embedding is defined as the average of the encoder outputs across all input tokens)
 - pretrain on large general corpus of QA pairs, then finetune on MSMARCO
 - only needed 10% of MSMARCO data to reach best results on BEIR
 - mean pooling:
 ```
 def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
 ```
 - using `T5Encoder`
 ```
 from transformers import T5Tokenizer, T5EncoderModel

 tokenizer = T5Tokenizer.from_pretrained("t5-small")
 model = T5EncoderModel.from_pretrained("t5-small")
 input_ids = tokenizer(
     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
 ).input_ids  # Batch size 1
 outputs = model(input_ids=input_ids)
 last_hidden_states = outputs.last_hidden_state
 ```
 12. Splade V2/++
- [SPLADE ++](https://arxiv.org/abs/2205.04733)
- Sparse models
- Use MLM + modern IR training techniques to build a sparse retrieval model 
- Github repo: https://github.com/naver/splade
- huggingface models: https://huggingface.co/naver
- Best sparse method on BEIR atm

13. SGPT: GPT Sentence Embeddings for Semantic Search
- Arxiv: https://arxiv.org/pdf/2202.08904v4.pdf
- github: https://github.com/muennighoff/sgpt
 - includes example code for bi-encoder/cross-encoder usage
 - Bi-encoder asymetric is what we care about (query and docs are not interchangable)
- For bi-encoder, fine-tuned only the bias tensors (see [bitfit](https://arxiv.org/abs/2106.10199)) w/ contrastive fine-tuning
 - example code to only FT some parameters in a model: https://www.kaggle.com/code/heyytanay/train-ai4code-pytorch-bert-large-w-b?scriptVersionId=95570216&cellId=22
 - encoder models (like GTR-XXL) still outperform SGPT in fine-tuned setting
 - but SGPT outperforms in zero-shot over GTR-XXL
- For cross-encoder, just took GPT as is using log probability extraction
- BEIR Results:
 - Bi-encoder: **0.490 average nDCG@10 on BEIR** (Current SOTA)

14. RetroMAE:
- https://arxiv.org/abs/2205.12035
- Results on MS MARCO:
 - MRR@10: 0.3501
  - better than laprador and condenser
  - no comparison to CoCondenser, ColbertV2 etc. because no nDCG@10
- BEIR Results:
 - avg **0.448 NDCG@10** (Current best for bert sized model eg no SGPT, GTR etc.)
 - pretty competitive especially at the size (BERT-base size)
- Novel pre-training framework for dense retrieval, 3 main designs
  1. Masked auto-encoding: input sentence is corrupted twice with 2 different masked (1 for encoder and 1 for decoder)
  2. Asymmetric structure: BERT base for encoder, 1 layer transformer decoder
  3. Asymmetric masking ratio: 15% for encoder, 50-90% ratio for decoder
  
15. Masked Autoencoders As The Unified Learners For Pre-Trained Sentence Representation
- Followup paper on RetroMAE
- https://arxiv.org/abs/2208.00231
- retroMAE on generic corpus has strong zero-shot performance (0.452 NDCG@10, just a 0.01 pts worse than GTR-XL)
- 2nd stage in domain pretraining on MS-MARCO (+ ANCE finetuning) yields better in domain performance on MSMARCO than CoCondenser (MRR@k)
 - but slightly worse out of domain performance on BEIR
- Can also achieve strong sentence embedding tasks


16. No Parameter Left Behind: How Distillation and Model Size Affect Zero-Shot Retrieval
- https://arxiv.org/abs/2206.02873
- twitter thread: https://twitter.com/rodrigfnogueira/status/1534564449452969985
- CE reranker latency issues are not as much of an issue when the initial number of retrieved document is small (achieve <1s latency with 3B param reranker with 50 retrieved docs from BM25 with better nccg@10 and only 100ms slower than a 22M param minilm reranker w/ 1000 retrieved docs)
- scaling model size has marginal effects on in domain performance, but affects generalization to OOD data to a much higher degree
- similar sized rerankers outperform dense retrievers, particularly zero-shot generalization (not that surprising)

17. Questions Are All You Need to Train a Dense Passage Retriever
- https://arxiv.org/abs/2206.10658
- Only need set of questions and a set of documents, no direct label for question/doc pairs needed
- Gets SOTA on several supervised benchmarks for **retrieval accuracy**
 - SQUAD-open
 - TriviaQA
 - NQ-Open
 - WebQ
- Also gets improvements (and some SOTA results) in zero shot transfer (no BEIR though)
 - Finetune on NQ-Open/MS MARCO/NQ-full + MS MARCO
 - evaluate on 
  - SQUAD-open
  - TriviaQA
  - WebQ
  - EW

18. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling (TAS-B) 
 - https://arxiv.org/abs/2104.06967
 - closely related to GPL, CE distillation (MarginMSE)
 - training code https://github.com/sebastian-hofstaetter/matchmaker

19. Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated Neural Text Retrievers
 - https://arxiv.org/abs/2207.07087
 - promp tuning instead full finetuning from a model checkpoint 
 - pt instead of ft on dpr-nq -> ms marco -> better performance on BEIR than full finetuning
 
20. Promptagator: Few-shot Dense Retrieval From 8 Examples
 - https://arxiv.org/abs/2209.11755
 - similar to GPL, but scaling super hard!
 - 3 main steps:
  1. Using a PLM (in this case 137B FLAN model) to generate queries (like QGen, GPL)
   - done either 0-shot or few-shot (up to 8 examples depending on context length)
  2. Consistency filtering using only generated data
   - methods like GPL require an external model as the filter (eg MSMARCO MiniLM)
   - Train *initial* retriever using ALL data
   - use initial retriever to filter out synthetic `(q, d)` pairs where `d` is not in top-k most retrieved docs (practically, `k=1`)
   - continue training from initial retriever checkpoint
   - performs better on 8/11 tested BEIR datasets
  3. Training retriever
   - start form T5 encoder checkpoint
   - pretrain on C4 with independent cropping task from Contriever
   - finetune using DPR on ALL data
   - After a set of epochs, apply filtering on synthetic data
   - Continue training on filtered data
   - When number of docs < 6k, use `bs=128`, otherise use `bs=6k`
  - Better than GPL on 8 measured tasks
  - better reranker too than monoT5 (3b) with a 110M model
  - Possible extension to include pseudolabel distillation:
   - use roudabout filtering to filter for relevancy
   - train CE reranker
   - use CE scores to distill via MarginMSE

## Training State-of-the-art Text Embedding Models from Sentence Transformers
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


## IR/Neural Search for low resource scenarios
https://twitter.com/Nils_Reimers/status/1452931505995624451?t=KYJjn4zKvjxaeCtnZgHWPg&s=09&fbclid=IwAR1dp1_QLuPabCnTF4lxWw8LSzVm5IsdGEPomhFYu7J5zjFqlA2_BeXKdlA

video: https://www.youtube.com/watch?v=XNJThigyvos



## Domain adaptation (from pretrained model)
https://www.youtube.com/watch?v=qzQPbIcQu9Q

problem: 
- IR models perform worse out of domain (eg model finetuned on MS-MARCO performs worse than BM25 on BioASQ)
- Due to text it has not seen before
- How do you adapt a model to out of domain data (**without any labelled data in the target domain**)

2 solutions
1. Adaptive pre-training
  - pre-train model on target domain (eg MLM, coCondenser, SimCSE, TSDAE etc.)
  - Fine-tune on labeled data **it was originally finetuned on ** (eg MS MARCO)
  - Note: this would also work if you have labeled data in target domain (eg finetuning on BioASQ). this just becomes a standard 2 stage finetuning task
  - However, this is expensive: requires pre-training AND finetuning
2. Unsupervised domain adaptation
  - Take a pre-trained model finetuned on a dataset (eg MS MARCO checkpoint)
  - unsupervised training on target domain
  - pros: don't have to run fine-tuning again (which is expensive)
  - How: use GPL (see below)


### Generative Pseudo Labeling (GPL) 
GPL is an unsupervised domain adaptation method for training dense retrievers. It is based on query generation and pseudo labeling with powerful cross-encoders. To train a domain-adapted model, it needs only the unlabeled target corpus and can achieve significant improvement over zero-shot models.

https://sbert.net/examples/domain_adaptation/README.html#gpl-generative-pseudo-labeling

code https://github.com/UKPLab/gpl

How GPL works:
- take a fine-tuned model and perform unsupervised domain adaptation
- 4 stages

1. Query generation
  - use T5 (currently best text generation model that can be fine-tuned)
  - either take a pre-trained checkpoint (eg finetuned on MS MARCO) or finetune on your own data if you have data (not directly finetuning on the target dataset, since it would not have query/doc pairs)
  - similar to doc2query (given doc, generate `n` queries about it)

2. Negative mining
  - use the generated questions to query the corpus to generate negative examples
  - eg given `original_doc, generated_query_1` search in ES using `generated_query_1` to get `k` docs (and exclude `original_doc` to get `k-1` negative docs)
  - end up with `k` `(query, doc)` pairs 1 positive and k-1 negatives

3. CE pseudo-labelling
  - score pairs with CE
  - use concatenated `(query, doc_1), ... (query_doc_k)` and CE to generate a pseudo-label score (how relevant the query and doc are)

4. Train bi-encoder with `MarginMSELoss`
  - can use either a fine-tuned model (eg finetuned on MS MARCO) or a base model (eg `bert-base-uncased`)
  - from GPL experiments: both reach about the same end result, but fine-tuned model takes shorter time (already have a head start)

**Note:**
- steps 3 and 4 can also be replaced with other training (eg dpr, cocondenser etc.) since you have query, positive/negative docs
 
 
## Synthetic Data Generation to Improve Model Performance
https://hackmd.io/gmDAH0fqRAKcZl3sPLdjsg

## Building a Search-Based Discord Bot with Language Models
https://txt.cohere.ai/building-a-search-based-discord-bot-with-language-models/?fbclid=IwAR1-PzwBsNYI8IYqGl2mmxYolNGJUu_CEj75XTWklbyU0_-Zhbe78s9uDlc

