# BiEncoder Architecture Project

A biomedical passage retrieval system built around a bi-encoder trained with contrastive
learning, indexed with FAISS, evaluated with retrieval metrics, and improved via
cross-encoder hard-negative mining.

## Pipeline

```
Ingestion: ClinicalTrials.gov API v2 -> paginate -> data/raw/studies.jsonl (~5k studies)
Training:  BERT (Q) + BERT (P) -> embeddings -> contrastive loss -> backprop
Indexing:  passages -> fine-tuned BERT -> embeddings -> FAISS index
Retrieval: query -> BERT -> query embedding -> FAISS -> top-K passages
Reranking: top-K candidates -> cross-encoder -> hard negatives -> retrain bi-encoder
```

## Repo layout

```
biencoder/
├── config.yaml                     # all hyperparameters / paths in one place
├── data/
│   ├── raw/                        # studies.jsonl (from the API) + fetch_state.json
│   └── processed/                  # tokenized / cached tensors
├── src/
│   ├── data/
│   │   ├── clinical_trials_api.py  # API pagination logic -> writes data/raw/studies.jsonl
│   │   ├── dataset.py              # PyTorch Dataset/DataLoader for (Q, P) pairs
│   │   └── preprocessing.py        # cleaning, tokenizing, train/val/test split
│   │                                 (reads data/raw/studies.jsonl; never calls the API)
│   ├── models/
│   │   ├── custom_encoder.py       # from-scratch transformer encoder (learning path)
│   │   ├── biencoder.py            # wraps Q-encoder + P-encoder, does pooling
│   │   └── cross_encoder.py        # joint query+passage scorer for reranking
│   ├── losses.py                   # InfoNCE / contrastive loss over in-batch negatives
│   ├── train.py                    # training loop: forward -> loss -> backprop -> AdamW
│   ├── index/
│   │   └── faiss_index.py          # build/save/load FAISS index over passage embeddings
│   ├── retrieval.py                # embed query -> FAISS search -> top-K passages
│   ├── evaluation/
│   │   └── metrics.py              # Recall@K, MRR, NDCG@K, latency
│   ├── mining/
│   │   └── hard_negative_mining.py # cross-encoder based hard negative extraction
│   └── utils.py                    # seeding, logging, checkpointing helpers
├── scripts/
│   ├── fetch_clinical_trials.py    # CLI entry point: API pagination -> data/raw/studies.jsonl
│   ├── run_training.py             # CLI entry point to train the bi-encoder
│   ├── build_faiss_index.py        # CLI entry point to embed corpus + build index
│   ├── run_evaluation.py           # CLI entry point to compute retrieval metrics
│   └── run_hard_negative_mining.py # CLI entry point for the mining + retrain loop
└── tests/
    ├── test_losses.py              # sanity checks on contrastive loss math
    └── test_metrics.py             # sanity checks on Recall@K/MRR/NDCG implementations
```

## Suggested build order

0. `src/data/clinical_trials_api.py` (run via `scripts/fetch_clinical_trials.py`) — pull
   ~5k studies from the ClinicalTrials.gov API into `data/raw/studies.jsonl`. This is
   the only module that talks to the network; everything below only ever reads that file.
1. `src/data/preprocessing.py` + `src/data/dataset.py` — get (Q, P) pairs loadable.
2. `src/models/biencoder.py` (start by wrapping HuggingFace BERT; `custom_encoder.py` is
   optional if you want to build the transformer from scratch to cement understanding).
3. `src/losses.py` + `src/train.py` — get contrastive training running on a small batch.
4. `src/index/faiss_index.py` + `src/retrieval.py` — embed the corpus, build the index, query it.
5. `src/evaluation/metrics.py` — measure Recall@K / MRR / NDCG on a labeled eval set.
6. `src/models/cross_encoder.py` + `src/mining/hard_negative_mining.py` — close the loop by
   mining hard negatives and retraining the bi-encoder.
