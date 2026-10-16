"""
custom_tokenizer.py  (OPTIONAL — build this if you want to learn tokenization from scratch)

WHY YOU'D WANT THIS:
Right now every module (preprocessing.py, dataset.py, train.py, retrieval.py) calls
`AutoTokenizer.from_pretrained(base_model_name)` and treats it as a black box: text in,
token IDs + attention mask out. This file lets you replace that black box with your own
implementation while keeping the exact same interface, so nothing else in the codebase
has to change.

WHAT TO WRITE HERE (pick ONE approach, roughly in order of difficulty):
  1. Whitespace/word-level tokenizer: split on whitespace + punctuation, build a
     vocab of the top-N most frequent words from your corpus, map OOV words to <unk>.
  2. Byte Pair Encoding (BPE) from scratch: start with a character-level vocab, greedily
     merge the most frequent adjacent symbol pairs until you hit `vocab_size` (this is
     what GPT-2's tokenizer does, roughly).
  3. WordPiece (what BERT actually uses): similar to BPE but picks merges that maximize
     likelihood of the training corpus rather than just raw frequency, and marks
     word-continuation pieces with a "##" prefix (e.g. "unhappiness" -> ["un", "##happy",
     "##ness"]).

Either way you need:
  - A vocab: token string -> integer ID (and the reverse mapping for decoding).
  - Special tokens: [PAD], [UNK], [CLS], [SEP] (mirror BERT's set so pooling code that
    assumes a leading [CLS]/no leading token still works consistently).
  - A `train()` step that builds the vocab from a list of raw texts (your corpus, e.g.
    data/raw/studies.jsonl passages) — this is the "fit" step, done once, offline.
  - An `encode()` step that maps arbitrary text -> token IDs using the fitted vocab,
    with truncation to `max_seq_length` and padding to the longest sequence in a batch.

INTERFACE CONTRACT (so this is a drop-in swap for AutoTokenizer everywhere else):
  tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    -> {"input_ids": LongTensor[B, L], "attention_mask": LongTensor[B, L]}
  This is exactly what preprocessing.py's tokenize_batch(), dataset.py's
  QueryPassageDataset, and retrieval.py's retrieve() already call — implement `__call__`
  to match and nothing downstream needs to know which tokenizer is in use.

HOW TO WIRE IT IN ONCE IMPLEMENTED:
  - Add to config.yaml under `model`: `use_custom_tokenizer: true` (and maybe
    `tokenizer_vocab_path` / `tokenizer_vocab_size`).
  - In train.py / wherever you currently do
      tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name"])
    branch on the flag:
      if config["model"]["use_custom_tokenizer"]:
          tokenizer = CustomTokenizer.load(config["model"]["tokenizer_vocab_path"])
      else:
          tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_name"])
  - If you pair this with custom_encoder.py, pass `tokenizer.vocab_size` as `dict_size`
    into `CustomEncoder(...)` instead of hardcoding it.
"""

import json
from collections import Counter
from pathlib import Path

import torch

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


class CustomTokenizer:
    """From-scratch tokenizer: text -> token IDs, with an AutoTokenizer-compatible API."""

    def __init__(self, vocab: dict | None = None):
        """
        vocab: token string -> int ID, must include SPECIAL_TOKENS if provided.
        TODO: if vocab is None, start empty (train() will populate it later).
        Set self.vocab, self.inverse_vocab (id -> token), and cache special token ids
        (self.pad_token_id, self.unk_token_id, self.cls_token_id, self.sep_token_id).
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """TODO: return len(self.vocab). Needed by CustomEncoder's `dict_size` arg."""
        raise NotImplementedError

    # ---- "fit" step: build the vocab from your corpus ----

    def train(self, texts: list, vocab_size: int = 8000, min_frequency: int = 2):
        """Build the vocab from a list of raw training texts.

        TODO: implement one of:
          - word-level: Counter over whitespace-split (lowercased) tokens, keep the
            `vocab_size - len(SPECIAL_TOKENS)` most frequent tokens above min_frequency.
          - BPE/WordPiece: start from characters, iteratively merge most-frequent pairs
            until vocab reaches `vocab_size`.
        Populate self.vocab / self.inverse_vocab (see __init__ TODO) when done.
        """
        raise NotImplementedError

    # ---- single-text encode/decode ----

    def _tokenize(self, text: str) -> list:
        """Split raw text into a list of token strings (before ID lookup).

        TODO: implement the splitting rule matching whatever train() uses
        (e.g. lowercase + whitespace/punctuation split, or greedy longest-match
        over the learned BPE/WordPiece vocab).
        """
        raise NotImplementedError

    def encode(self, text: str, max_length: int = 256, truncation: bool = True) -> list:
        """Text -> list[int] token IDs, with [CLS] ... [SEP] added, OOV -> [UNK].

        TODO: implement:
          1. tokens = self._tokenize(text)
          2. ids = [self.cls_token_id] + [self.vocab.get(t, self.unk_token_id) for t in tokens] + [self.sep_token_id]
          3. if truncation and len(ids) > max_length: ids = ids[:max_length-1] + [self.sep_token_id]
          4. return ids
        """
        raise NotImplementedError

    def decode(self, ids: list) -> str:
        """List[int] token IDs -> text (mainly for debugging/sanity checks).

        TODO: implement: map ids back through self.inverse_vocab, skip special tokens,
        join with spaces (or "".join for BPE, stripping "##"/continuation markers).
        """
        raise NotImplementedError

    # ---- batch call, matching the HF AutoTokenizer interface ----

    def __call__(
        self,
        texts,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
        return_tensors: str = "pt",
    ) -> dict:
        """Batch-encode texts. Must match AutoTokenizer's call signature/output shape
        so this is a drop-in swap in preprocessing.py / dataset.py / retrieval.py.

        TODO: implement:
          1. if isinstance(texts, str): texts = [texts]
          2. all_ids = [self.encode(t, max_length, truncation) for t in texts]
          3. if padding: pad every sequence up to max(len(ids) for ids in all_ids)
             with self.pad_token_id, and build a parallel attention_mask
             (1 for real tokens, 0 for padding).
          4. if return_tensors == "pt": wrap input_ids/attention_mask in torch.tensor(...)
          5. return {"input_ids": ..., "attention_mask": ...}
        """
        raise NotImplementedError

    # ---- persistence, so you only train() once and reuse it everywhere ----

    def save(self, path: str) -> None:
        """Save the vocab (+ config) to disk as JSON so train.py/retrieval.py can reload
        without retraining.

        TODO: implement, e.g.:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            json.dump({"vocab": self.vocab}, open(path, "w"))
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "CustomTokenizer":
        """Load a previously trained tokenizer's vocab from disk.

        TODO: implement, e.g.:
            data = json.load(open(path))
            return cls(vocab=data["vocab"])
        """
        raise NotImplementedError
