"""
metrics.py

WHAT TO WRITE HERE (per your notes' "Evaluation" section):

- `recall_at_k(retrieved_ids, relevant_ids, k) -> float`:
    Of all relevant passages for a query, what fraction appear in the top-K retrieved?
    recall@k = |retrieved_ids[:k] ∩ relevant_ids| / |relevant_ids|
    Compute for k in config.yaml's `k_values` (e.g. [5, 10, 20]), averaged over all
    eval queries. This is the "critical" metric per your notes.

- `mean_reciprocal_rank(retrieved_ids_per_query, relevant_ids_per_query) -> float`:
    For each query, find the rank (1-indexed) of the FIRST relevant passage in the
    retrieved list; reciprocal_rank = 1 / rank (0 if none found within the retrieved
    list). MRR = average reciprocal_rank across all queries. Measures ranking quality
    ("how high is the first relevant result?").

- `ndcg_at_k(retrieved_ids, relevance_grades, k) -> float`:
    For graded relevance (not just binary relevant/irrelevant — e.g. irrelevant=0,
    relevant=1, highly relevant=2), compute:
      DCG@k = sum_{i=1}^{k} relevance(retrieved_ids[i]) / log2(i + 1)
      IDCG@k = DCG@k of the ideal ordering (relevances sorted descending)
      NDCG@k = DCG@k / IDCG@k
    Particularly appropriate for a manually labeled biomedical benchmark per your notes,
    since it can express "somewhat relevant" vs "highly relevant" passages.

- (Optional) `latency_ms(retrieval_fn, queries) -> float`:
    Measure average ms/query for retrieval, useful to compare BM25 vs bi-encoder vs
    bi-encoder+reranking.

Each function should work on a single query's results and be combined via a wrapper
(e.g. `evaluate_all(...)` in evaluate.py or here) that averages over the full eval set.
"""

from typing import List, Dict
import numpy as np


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """TODO: implement per formula above."""
    raise NotImplementedError


def mean_reciprocal_rank(
    retrieved_ids_per_query: List[List[str]],
    relevant_ids_per_query: List[List[str]],
) -> float:
    """TODO: implement per formula above."""
    raise NotImplementedError


def ndcg_at_k(retrieved_ids: List[str], relevance_grades: Dict[str, float], k: int) -> float:
    """TODO: implement DCG/IDCG per formula above.

    `relevance_grades` maps passage_id -> graded relevance (0 = irrelevant, higher = more
    relevant); passages not present default to 0.
    """
    raise NotImplementedError


def latency_ms(retrieval_fn, queries: List[str]) -> float:
    """TODO: implement, e.g. time.perf_counter() around retrieval_fn(query) per query,
    return average milliseconds per query.
    """
    raise NotImplementedError
