"""
test_losses.py

WHAT TO WRITE HERE:
Sanity checks for src/losses.py's contrastive_loss:
- If query_embeddings == passage_embeddings exactly (perfect positives, diagonal = 1,
  off-diagonal < 1), loss should be small/near its minimum.
- If embeddings are random/orthogonal, loss should be close to log(batch_size) (the
  "chance level" cross-entropy for B classes), a useful numerical sanity check.
- Loss should be a scalar, finite, and require grad (so backward() works).
- Test that batch size B=1 behaves sanely (no negatives — check it doesn't crash).

TODO: implement these tests using pytest once contrastive_loss is implemented.
"""

import math
import torch
import pytest


def test_placeholder():
    # TODO: replace with real tests once src/losses.py is implemented, e.g.:
    #
    # from src.losses import contrastive_loss
    # embeds = torch.eye(4)  # perfectly separated "embeddings"
    # loss = contrastive_loss(embeds, embeds, temperature=0.05)
    # assert loss.item() < 0.1
    pytest.skip("Implement src/losses.py first, then fill in real assertions here.")
