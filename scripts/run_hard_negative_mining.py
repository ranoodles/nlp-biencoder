"""
run_hard_negative_mining.py

WHAT TO WRITE HERE:
CLI entry point for the hard-negative-mining + retrain loop, per your notes' Reranking
section (Bi-encoder -> Top 100 -> Cross-Encoder -> Labeling -> Retrain Bi-Encoder).

TODO:
1. Load config, load the trained bi-encoder + FAISS index (as a Retriever), and load/
   instantiate the CrossEncoder (src/models/cross_encoder.py).
2. Load the set of training queries to mine over (could be your existing training
   queries, or a broader query set if available).
3. Call src/mining/hard_negative_mining.py's mine_hard_negatives(...) to produce a new
   dataset of {"query", "positive", "hard_negatives"} records.
4. Save the mined dataset to data/processed/ (e.g. hard_negatives.jsonl).
5. Kick off retraining: either call src/train.py's train() with a config flag pointing
   at the new hard-negatives dataset, or add a dedicated retrain function that uses the
   extended loss described in hard_negative_mining.py's docstring.
6. Optionally loop: after retraining, re-run run_evaluation.py, and if desired, mine
   again with the improved model for another round.

Example usage once implemented:
    python scripts/run_hard_negative_mining.py --config config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives and retrain.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # TODO: implement per the steps above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
