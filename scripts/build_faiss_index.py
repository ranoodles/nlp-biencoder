"""
build_faiss_index.py

WHAT TO WRITE HERE:
CLI entry point to embed the full passage corpus with the (trained) bi-encoder and
build/save a FAISS index, per your notes' FAISS section:
  50k passages -> fine-tuned BERT -> embeddings -> FAISS index

TODO:
1. Load config, load the corpus (src/data/preprocessing.py's load_corpus).
2. Load the trained BiEncoder model from a checkpoint (src/utils.py's load_checkpoint,
   or just load_state_dict directly).
3. `embeddings, id_map = embed_corpus(...)` (src/index/faiss_index.py).
4. `index = build_index(embeddings, config["faiss"]["index_type"])`.
5. `save_index(index, config["faiss"]["index_path"])`.
6. `save_id_map(id_map, config["faiss"]["id_map_path"])`.
7. Also consider saving the raw embeddings array (config["faiss"]["embeddings_path"])
   in case you want to rebuild a different index type later without re-embedding.

Example usage once implemented:
    python scripts/build_faiss_index.py --config config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Embed corpus and build FAISS index.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # TODO: implement per the steps above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
