"""
hard_negative_mining.py

WHAT TO WRITE HERE (per your notes' "Reranking" / "Hard Negative Mining" section):

  Query -> Bi-encoder (Retrieval) -> Top 100 Candidates -> Cross-Encoder (Precise)
        -> Labeling & Filtering -> Retrain Bi-Encoder

Implement the 4-step loop from your notes:

1. `get_candidates(query, retriever, top_k_candidates) -> list[passage]`:
   Use the CURRENT bi-encoder (src/retrieval.py's retrieve()) to get the top-K
   (e.g. 100) candidate passages for a training query.

2. `label_candidates(query, candidates, cross_encoder, pos_threshold, neg_threshold) -> dict`:
   Score every candidate with the cross-encoder (src/models/cross_encoder.py). Bucket
   each into:
     - "true_positives": score >= pos_threshold (config.yaml's `positive_score_threshold`)
     - "hard_negatives": score <= neg_threshold (config.yaml's `negative_score_threshold`)
     - "danger_zone": pos_threshold > score > neg_threshold -> EXCLUDE from training,
       since these might be valid alternative answers the bi-encoder correctly found but
       weren't in your original labeled positive set. Do not penalize the model for these.

3. `mine_hard_negatives(queries, retriever, cross_encoder, config) -> list[dict]`:
   Loop over a set of training queries, run steps 1-2 for each, and build a new dataset
   of {"query": ..., "positive": ..., "hard_negatives": [...]} records. Save this to
   data/processed/ (e.g. as JSONL) for use in retraining.

4. Retraining: extend src/losses.py / src/train.py (or add a variant here) so that
   instead of relying solely on in-batch negatives, the loss also includes the explicit
   mined hard negatives per query (common approaches: add them as extra columns in the
   similarity matrix, or use a margin-based loss / Margin MSE against the cross-encoder's
   scores, or InfoNCE with explicit negatives appended). Document which approach you pick.

This module is invoked by scripts/run_hard_negative_mining.py, likely as a repeatable
step: mine -> retrain -> re-evaluate -> (optionally) mine again with the improved model.
"""

from typing import List, Dict


def get_candidates(query: str, retriever, top_k_candidates: int) -> List[dict]:
    """TODO: implement using src/retrieval.py's retrieve()."""
    raise NotImplementedError


def label_candidates(
    query: str,
    candidates: List[dict],
    cross_encoder,
    pos_threshold: float,
    neg_threshold: float,
) -> Dict[str, list]:
    """TODO: implement scoring + bucketing per the 3-way split described above.

    Returns e.g. {"true_positives": [...], "hard_negatives": [...], "danger_zone": [...]}.
    """
    raise NotImplementedError


def mine_hard_negatives(queries: List[str], retriever, cross_encoder, config: dict) -> List[dict]:
    """TODO: implement the full mining loop over `queries`, returning training records:
    [{"query": ..., "positive": ..., "hard_negatives": [...]}, ...]
    """
    raise NotImplementedError
