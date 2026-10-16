"""
faiss_index.py

WHAT TO WRITE HERE (per your notes' "FAISS" section):
  50,000 passages -> fine-tuned BERT -> 50,000 embeddings -> FAISS index

Steps to implement:
1. `embed_corpus(model, corpus, tokenizer, batch_size, device) -> np.ndarray`:
   run every passage in the corpus through the trained BiEncoder's `encode_passage`
   in batches (no_grad / eval mode), stack into a single [N, D] numpy array. Also keep
   a parallel list/dict mapping FAISS row index -> original passage id (needed since
   FAISS only returns integer indices, not your own IDs).
2. `build_index(embeddings: np.ndarray, index_type: str) -> faiss.Index`:
   - "flat": `faiss.IndexFlatIP(d)` (exact inner-product search; since embeddings are
     L2-normalized, inner product == cosine similarity) — fine for up to ~1M vectors.
   - "ivf": `faiss.IndexIVFFlat` — faster approximate search for larger corpora, needs
     `index.train(embeddings)` before adding vectors.
   - "hnsw": `faiss.IndexHNSWFlat` — good latency/accuracy tradeoff, no training needed.
   Then `index.add(embeddings)`.
3. `save_index(index, path)` / `load_index(path)`: use `faiss.write_index` /
   `faiss.read_index`.
4. `save_id_map(id_map, path)` / `load_id_map(path)`: persist the row-index -> passage-id
   mapping (e.g. as JSON) alongside the index so retrieval.py can translate results back
   to real passage IDs/text.

This module is used by scripts/build_faiss_index.py to do a one-time (or periodic) index
build after training/fine-tuning the bi-encoder.
"""

import json
import numpy as np


def embed_corpus(model, corpus, tokenizer, batch_size: int, device) -> np.ndarray:
    """Embed every passage in `corpus` using the trained bi-encoder's passage encoder.

    TODO: implement in eval mode with torch.no_grad(), batching for memory efficiency.
    Returns an [N, D] float32 numpy array (L2-normalized rows).
    """
    raise NotImplementedError


def build_index(embeddings: np.ndarray, index_type: str = "flat"):
    """Build a FAISS index over passage embeddings.

    TODO: implement flat / ivf / hnsw variants as described above.
    """
    raise NotImplementedError


def save_index(index, path: str) -> None:
    """TODO: implement via faiss.write_index(index, path)."""
    raise NotImplementedError


def load_index(path: str):
    """TODO: implement via faiss.read_index(path)."""
    raise NotImplementedError


def save_id_map(id_map: dict, path: str) -> None:
    """TODO: implement, e.g. json.dump(id_map, open(path, 'w'))."""
    raise NotImplementedError


def load_id_map(path: str) -> dict:
    """TODO: implement, e.g. json.load(open(path)). Remember JSON keys become strings."""
    raise NotImplementedError
