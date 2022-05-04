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

4. SBERT Models:
- `multi-qa-mpnet-base-dot-v1`
- `msmarco-bert-base-dot-v5`

5. Combining dense embeddings with sparse models (BM25/Splade v2 etc)
- splade v2
- combining sparse and dense models: 2 runs
  - 1 run with sparse retrieval (eg BM25) and 1 run with dense (dpr/cocondenser etc.)
  - linear combination of the 2 scores as the new score (similar to score combiner work)

6. LaPraDoR
- https://arxiv.org/abs/2203.06169
- unsupervised
- SOTA in zero-shot retrieval (BEIR)
- claim: doesn't need to fine-tune on a downstream dataset

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


10. Unsupervised Passage Re-ranker (UPR)
- https://arxiv.org/abs/2204.07496
- twitter thread: https://twitter.com/Devendr06654102/status/1516106999175467018?fbclid=IwAR3C78yR4NYN9KjPTo5EDpdNhdKSS0-0hHb6hWg_VgRI3xidvT9t9gnQZLQ
- on top of a model (eg DPR)


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
  - either take a pre-trained checkpoint (eg finetuned on MS MARCO) or finetune on your own data if you have data
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
