"""
run_evaluation.py

WHAT TO WRITE HERE:
CLI entry point to compute Recall@K / MRR / NDCG@K on your labeled biomedical eval set.

TODO:
1. Load config, load the FAISS index + id_map (src/index/faiss_index.py's load_index/
   load_id_map), load the trained bi-encoder.
2. Load your eval labels file (query -> list of relevant passage ids, or graded
   relevance for NDCG) via src/data/preprocessing.py.
3. For each eval query: run src/retrieval.py's retrieve() to get top-K passage ids.
4. Accumulate retrieved_ids_per_query + relevant_ids_per_query.
5. Compute recall_at_k for each k in config["evaluation"]["k_values"], mean_reciprocal_rank,
   and ndcg_at_k (src/evaluation/metrics.py), averaged over all eval queries.
6. Print/log a results table; optionally save to a JSON/CSV report file for tracking
   improvement across bi-encoder vs bi-encoder+hard-negative-retrain vs +reranking.

Example usage once implemented:
    python scripts/run_evaluation.py --config config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # TODO: implement per the steps above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
