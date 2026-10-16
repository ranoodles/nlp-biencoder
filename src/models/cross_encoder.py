"""
cross_encoder.py

WHAT TO WRITE HERE (per your notes' "Reranking" section):
- Unlike the bi-encoder (which encodes query and passage SEPARATELY then compares
  embeddings), a cross-encoder feeds the query and passage TOGETHER as one input
  (e.g. "[CLS] query [SEP] passage [SEP]") through a transformer, and outputs a single
  relevance score. This is slower (can't precompute passage embeddings / can't use
  FAISS) but much more accurate — hence it's used as a "teacher" for reranking a small
  candidate set (top 100), not for searching the full corpus.
- Load a pretrained cross-encoder (e.g. HuggingFace `AutoModelForSequenceClassification`
  with a model like "cross-encoder/ms-marco-MiniLM-L-6-v2") OR fine-tune your own.
- Implement `score(query: str, passage: str) -> float` and a batched
  `score_batch(query_passage_pairs: list[tuple[str, str]]) -> list[float]`.
- This class is used in two places:
    1. src/mining/hard_negative_mining.py — to label bi-encoder's top-K candidates as
       true positives / hard negatives / danger-zone (excluded).
    2. Optionally in src/retrieval.py as a final reranking step over FAISS's top-K
       output before returning results to the user.
"""

import torch
import torch.nn as nn


class CrossEncoder(nn.Module):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        # TODO: load AutoTokenizer + AutoModelForSequenceClassification from model_name.
        raise NotImplementedError

    def score(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair for relevance.

        TODO: tokenize the pair jointly, run through the model, return a scalar score
        (e.g. sigmoid of the logit, or raw logit depending on the model).
        """
        raise NotImplementedError

    def score_batch(self, pairs) -> list:
        """Score a batch of (query, passage) pairs. pairs: list[tuple[str, str]].

        TODO: implement batched tokenization + forward pass for efficiency, since this
        will be called on up to `top_k_candidates` (e.g. 100) passages per query during
        hard negative mining.
        """
        raise NotImplementedError
