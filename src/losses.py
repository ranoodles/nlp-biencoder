"""
losses.py

WHAT TO WRITE HERE (per your notes' "Backprop Architecture" / "Contrastive" section):
  similarity scores -> contrastive loss -> backpropagation -> gradients -> AdamW
                     -> update transformer weights

Given a batch of B queries and B positive passages:
  1. Compute the [B, B] similarity matrix: sim[i][j] = cosine_similarity(query_i, passage_j)
     (if embeddings are already L2-normalized in biencoder.py, this is just query @ passage.T).
  2. Divide by a temperature (config.yaml's `temperature`, e.g. 0.05) to sharpen the
     distribution before softmax — smaller temperature = more confident distinctions.
  3. The diagonal (sim[i][i]) is the positive pair; every off-diagonal element sim[i][j]
     for j != i is an "in-batch negative" (per your notes' example table with Q1..Q4/P1..P4).
  4. Apply softmax across each row -> this is InfoNCE / contrastive loss: it maximizes the
     positive pair's probability while minimizing all negatives' probability in the same step.
  5. Loss = cross-entropy between each row's softmax output and the correct index (i.e.
     `labels = torch.arange(B)`, since positives sit on the diagonal).
  6. (Optional, symmetric loss) Do the same computation on the transposed matrix
     (passage-to-query direction) and average both directions — this is common in
     bi-encoder training (e.g. CLIP-style symmetric InfoNCE) and squeezes more signal
     out of each batch.

This function returns the scalar loss that train.py calls `.backward()` on.
"""

import torch
import torch.nn.functional as F


def contrastive_loss(
    query_embeddings: torch.Tensor,
    passage_embeddings: torch.Tensor,
    temperature: float = 0.05,
    symmetric: bool = True,
) -> torch.Tensor:
    """In-batch contrastive (InfoNCE) loss for a bi-encoder.

    Args:
        query_embeddings: [B, D] L2-normalized query embeddings.
        passage_embeddings: [B, D] L2-normalized passage embeddings, where row i is the
            positive passage for query i.
        temperature: softmax temperature.
        symmetric: if True, average query->passage and passage->query losses.

    TODO: implement per the steps described above.
    """
    norm_q = torch.softmax(query_embeddings, dim=-1)
    norm_p = torch.softmax(passage_embeddings, dim=-1)
    similarity_matrix = torch.matmul(norm_q, norm_p.T) / temperature
    similarity_matrix = torch.exp(similarity_matrix)
    pos_pair_scores = similarity_matrix.diagonal()
    row_sums = similarity_matrix.sum(dim=-1)
    loss = -torch.log(pos_pair_scores / row_sums)
    avg_loss = torch.mean(loss)
    return avg_loss
