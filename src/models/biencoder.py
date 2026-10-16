"""
biencoder.py

WHAT TO WRITE HERE (per your notes' "Use Existing Encoder" path — recommended starting point):
  Text -> BERT tokenizer -> Token IDs + attention mask -> BERT transformer
       -> Token representations -> Mean pooling -> Sentence embedding

This is the main model class tying everything together:
- Load a pretrained transformer via `AutoModel.from_pretrained(base_model_name)`
  (e.g. bert-base-uncased, or a biomedical variant like PubMedBERT) — OR, if
  config.yaml's `use_custom_encoder` is true, use `CustomEncoder` from custom_encoder.py.
- Implement a `mean_pooling(token_embeddings, attention_mask)` function: average the
  token representations, but only over real tokens (mask out padding) — this is the
  standard sentence-embedding trick since raw [CLS] isn't always the best summary vector.
- Optionally add a linear projection head down to `embedding_dim` (128) if you want a
  smaller final embedding size than BERT's native 768.
- IMPORTANT DESIGN DECISION per "bi-encoder": are the query encoder and passage encoder
  the SAME weights (shared/tied encoder, most common for symmetric retrieval) or two
  separate encoders? Decide and document it here. A shared encoder is simplest to start.
- Expose two methods: `encode_query(input_ids, attention_mask)` and
  `encode_passage(input_ids, attention_mask)` (they may call the same underlying encoder).
- Consider L2-normalizing output embeddings — makes cosine similarity == dot product,
  which simplifies both the contrastive loss and FAISS index type (inner-product index).
"""

import torch
import torch.nn as nn


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over non-padded tokens.

    TODO: implement. token_embeddings: [B, L, D], attention_mask: [B, L] -> returns [B, D].
    """
    raise NotImplementedError


class BiEncoder(nn.Module):
    """Bi-encoder wrapping a (shared or separate) transformer for query/passage encoding."""

    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        embedding_dim: int = 128,
        pooling: str = "mean",
        use_custom_encoder: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        # TODO: load AutoModel.from_pretrained(base_model_name) if not use_custom_encoder,
        # else instantiate CustomEncoder. Add an optional projection nn.Linear(hidden, embedding_dim).
        raise NotImplementedError

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized [batch_size, embedding_dim] query embeddings."""
        raise NotImplementedError

    def encode_passage(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized [batch_size, embedding_dim] passage embeddings."""
        raise NotImplementedError

    def forward(self, query_batch: dict, passage_batch: dict):
        """Convenience forward for training: returns (query_embeds, passage_embeds)."""
        raise NotImplementedError
