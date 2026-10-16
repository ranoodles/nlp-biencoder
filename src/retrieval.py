"""
retrieval.py

WHAT TO WRITE HERE (per your notes' retrieval-time flow):
  query -> BERT -> query embedding -> FAISS -> top-K passages

Steps to implement:
1. `retrieve(query: str, model, tokenizer, faiss_index, id_map, top_k, device) -> list`:
   - Tokenize the query text.
   - Run it through the bi-encoder's `encode_query` to get a [1, D] embedding
     (L2-normalized, no_grad, eval mode).
   - Call `faiss_index.search(query_embedding, top_k)` -> returns (distances, indices).
   - Map each returned FAISS row index back to a real passage id/text via `id_map`.
   - Return a ranked list of {"passage_id": ..., "text": ..., "score": ...} dicts.
2. (Optional) `retrieve_and_rerank(...)`: take the top-K (e.g. 100) from FAISS, then run
   them through the cross-encoder (src/models/cross_encoder.py) to re-sort by a more
   precise relevance score before returning the final top-N to the user. This mirrors
   the "Reranking" diagram in your notes (Bi-encoder -> Top 100 -> Cross-Encoder -> final).
3. Consider wrapping this in a small class (e.g. `Retriever`) that loads the model,
   index, and id_map once and exposes `.search(query, top_k)` for repeated use (e.g. in
   a demo script or simple API), rather than reloading everything per call.
"""

import numpy as np


def retrieve(query: str, model, tokenizer, faiss_index, id_map: dict, top_k: int, device):
    """Embed a query and search the FAISS index for the top-K passages.

    TODO: implement per the steps above.
    """
    raise NotImplementedError


def retrieve_and_rerank(
    query: str,
    model,
    tokenizer,
    faiss_index,
    id_map: dict,
    cross_encoder,
    top_k_candidates: int,
    top_n_final: int,
    device,
):
    """Retrieve top_k_candidates via FAISS, then rerank with a cross-encoder.

    TODO: implement: call retrieve() for candidates, score each (query, passage_text)
    pair with cross_encoder.score_batch(), sort descending, return top_n_final.
    """
    raise NotImplementedError
