"""
test_metrics.py

WHAT TO WRITE HERE:
Sanity checks for src/evaluation/metrics.py using hand-computed toy examples so you can
verify the math is right before trusting it on real eval data:

- recall_at_k: e.g. relevant_ids=["a","b","c"], retrieved_ids=["a","x","b","y","z"],
  k=5 -> recall should be 2/3.
- mean_reciprocal_rank: e.g. first relevant item at rank 3 -> reciprocal_rank = 1/3;
  average across a couple of queries with known ranks.
- ndcg_at_k: construct a small case with known relevance grades and verify against a
  hand-computed DCG/IDCG value (there are many worked examples online for NDCG@k to
  check against).

TODO: implement these tests using pytest once src/evaluation/metrics.py is implemented.
"""

import pytest


def test_placeholder():
    # TODO: replace with real tests once src/evaluation/metrics.py is implemented, e.g.:
    #
    # from src.evaluation.metrics import recall_at_k
    # retrieved = ["a", "x", "b", "y", "z"]
    # relevant = ["a", "b", "c"]
    # assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)
    pytest.skip("Implement src/evaluation/metrics.py first, then fill in real assertions here.")
