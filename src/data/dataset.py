"""
dataset.py

WHAT TO WRITE HERE:
- A `torch.utils.data.Dataset` subclass (e.g. `QueryPassageDataset`) that wraps the
  (query, passage) pairs produced by preprocessing.py and returns tokenized tensors
  for a query and its positive passage.
- A collate function for `DataLoader` that pads a batch of queries and a batch of
  passages separately (since Q and P go through the encoder(s) independently in a
  bi-encoder — that's the whole point vs. a cross-encoder).
- Note: you do NOT need to construct explicit negatives in the Dataset — in-batch
  negatives come for free from the other passages in the same batch during training
  (see src/losses.py). Only src/mining/hard_negative_mining.py adds explicit hard
  negatives later.
- Optionally, a separate Dataset for the eval set (query -> list of relevant passage ids)
  used by src/evaluation/metrics.py.

Key shapes to produce per batch:
  queries:  {input_ids: [B, L], attention_mask: [B, L]}
  passages: {input_ids: [B, L], attention_mask: [B, L]}
where row i of queries corresponds to row i of passages as the positive pair.
"""

import torch
from torch.utils.data import Dataset


class QueryPassageDataset(Dataset):
    """Dataset of positive (query, passage) pairs for contrastive training.

    TODO: implement __init__ (store tokenized data or raw pairs + tokenizer),
    __len__, and __getitem__ (return a single query/passage tokenized pair).
    """

    def __init__(self, pairs, tokenizer, max_seq_length: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


def collate_qp_batch(batch):
    """Collate a list of (query, passage) tokenized items into batched tensors.

    TODO: implement, e.g. stack/pad input_ids and attention_mask for queries and
    passages separately, return a dict: {"query": {...}, "passage": {...}}.
    """
    raise NotImplementedError
