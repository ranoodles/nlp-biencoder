"""
preprocessing.py

NOTE ON WHERE RAW DATA COMES FROM:
This module only ever READS from data/raw/ — it never calls the network. Fetching the
raw corpus from the ClinicalTrials.gov API is src/data/clinical_trials_api.py's job
(invoked via scripts/fetch_clinical_trials.py); it writes data/raw/studies.jsonl (one
JSON object per line: {"id": <NCT ID>, "text": ...}). Run that script first, then
build the functions below on top of its output.

WHAT TO WRITE HERE:
- Functions to load your raw (query, passage) pair data (e.g. from JSONL/CSV) and the
  full retrieval corpus (data/raw/studies.jsonl, produced by fetch_clinical_trials.py).
- Text cleaning (whitespace, casing, special chars) appropriate for biomedical text.
- Tokenization using the BERT tokenizer (`AutoTokenizer.from_pretrained(base_model_name)`):
  produce token IDs + attention masks, matching config.yaml's `max_seq_length`.
- A train/val/test split function (e.g. via sklearn's train_test_split), since your
  labeled eval set (for Recall@K/MRR/NDCG) should be held out and never trained on.
- A function to cache/save the processed tensors to `data/processed/` so preprocessing
  doesn't need to rerun every time you train (e.g. via torch.save or HF datasets .save_to_disk).

Key concepts to keep in mind while implementing (from your notes):
- Tokenization produces token IDs; an embedding is what a transformer converts those into.
- You need query text AND passage text side by side, associated as positive pairs.
"""

from pathlib import Path
from datetime import datetime
import json
# import unicodedata
# import re


def load_qp_pairs(studies_path: str, query_path: str):
    """Load (date, query, passage) positive pairs from disk.

    TODO: implement. Expected return: list of dicts like {"query": str, "passage": str}.
    """
    studies_temp = {}
    train = []
    test = []
    val = []
    with open(studies_path, "r", encoding="utf-8") as file:
      for line in file:
        a = json.loads(line.strip())
        studies_temp[a["id"]] = {"date": a["date"], "text": a["text"]}
    with open(query_path, "r", encoding="utf-8") as file:
      for line in file:
        b = json.loads(line.strip())
        obj = studies_temp.get(b["id"])
        pairs = [{"query": b["queries"][0], "passage": obj["text"]},
        {"query": b["queries"][1], "passage": obj["text"]}]
        if obj["date"][:4] <= 2021:
          train.extend(pairs)
        elif obj["date"][:4] <= 2023:
          val.extend(pairs)
        else:
          test.extend(pairs)
    return train, val, test


def load_corpus(path: str):
    """Load the full retrieval corpus (all passages to be indexed by FAISS).

    TODO: implement. `path` should point at data/raw/studies.jsonl produced by
    scripts/fetch_clinical_trials.py. Read it line by line with json.loads, since
    it's JSONL (one JSON object per line), not a single JSON array.
    Expected return: list of dicts like {"id": str, "text": str}.
    """
    raise NotImplementedError

# def clean_text(text: str) -> str:
#     """Basic text cleaning before tokenization.

#     - Strips leading/trailing whitespace
#     - Normalizes unicode
#     - Converts inner whitespace sequences to single spaces
#     - Removes control characters
#     """
#     text = unicodedata.normalize("NFKC", text)
#     text = text.strip()
#     text = re.sub(r'\s+', ' ', text)
#     text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\t\n\r')
#     return text


# def tokenize_batch(texts, tokenizer):
#     """Tokenize a batch of texts into input_ids + attention_mask tensors.

#     TODO: implement using `tokenizer(texts, padding=True, truncation=True,
#     max_length=max_seq_length, return_tensors="pt")`.
#     """
#     return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


# def split_train_val_test(pairs, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42):
#     """Split (query, passage) pairs into train/val/test sets.

#     TODO: implement, keep test set fully held out for final Recall@K/MRR/NDCG evaluation.
#     """
#     raise NotImplementedError


def save_processed(obj, path: str):
    """Cache a processed object (tokenized tensors, splits, etc.) to data/processed/.

    TODO: implement (e.g. torch.save(obj, path)).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError
