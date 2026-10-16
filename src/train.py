"""
train.py

WHAT TO WRITE HERE:
- The main training loop tying together dataset.py, biencoder.py, and losses.py.
- Steps per your notes' Backprop Architecture:
    for each batch of (query, passage) pairs:
        query_embeds, passage_embeds = model(query_batch, passage_batch)
        loss = contrastive_loss(query_embeds, passage_embeds, temperature)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
- Load hyperparameters from config.yaml (batch_size, learning_rate, num_epochs,
  temperature, warmup_steps).
- Use `torch.optim.AdamW` as the optimizer (per your notes).
- Optionally add an LR scheduler with warmup (`transformers.get_linear_schedule_with_warmup`).
- Log training loss per step/epoch (see src/utils.py for a logging helper).
- Save model checkpoints to config.yaml's `checkpoint_dir` (e.g. every epoch, or best-val).
- Optionally run periodic validation: encode val queries/passages, compute Recall@K
  using src/evaluation/metrics.py, and track the best checkpoint.
- This module should expose a `train(config)` function that scripts/run_training.py calls.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.preprocessing import load_qp_pairs
from datasets import Dataset
from src.models.custom_encoder import CustomEncoder
from src.losses import contrastive_loss
from src.utils import save_checkpoint, set_seed

# query_enc = self.tokenizer(
#           self.pairs[idx]["query"],
#           max_length=self.max_len,
#           padding='max_length',
#           truncation=True,
#           return_tensors="pt"
#       )

def tokenize_data(data, tokenizer, max_len):
  queries = data["query"]
  passages = data["passage"]
  queries = tokenizer(queries, truncation=True, padding="max_length", max_length=max_len)
  passages = tokenizer(passages, truncation=True, padding="max_length", max_length=max_len)
  return {
    "query_input_ids": queries["input_ids"],
    "query_attention_mask": queries["attention_mask"],
    "passage_input_ids": passages["input_ids"],
    "passage_attention_mask": passages["attention_mask"],
  }

def makeDataLoader(data, tokenizer, batch_size, max_len):
  dataset = Dataset.from_list(data)
  dataset = dataset.map(tokenize_data, fn_kwargs={"tokenizer": tokenizer, "max_len": max_len}, remove_columns=dataset.column_names, batched=True)
  dataset = dataset.with_format("torch")
  return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(config: dict):
    """Run the full bi-encoder training loop given a loaded config dict.

    TODO: implement:
      1. Set random seed (see src/utils.py).
      2. Load tokenizer + build train Dataset/DataLoader (src/data/).
      3. Instantiate BiEncoder model (src/models/biencoder.py) and move to device.
      4. Instantiate AdamW optimizer + optional scheduler.
      5. Loop over epochs/batches, compute contrastive_loss, backprop, step optimizer.
      6. Periodically log loss and (optionally) run validation retrieval metrics.
      7. Save checkpoints to config["training"]["checkpoint_dir"].
    """
    set_seed(config["training"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    dict_size = len(tokenizer)
    train, val, test = load_qp_pairs(config["data"]["studies_file"], config["data"]["queries_file"])
    train_loader = makeDataLoader(train, tokenizer, config["training"]["batch_size"], config["model"]["max_seq_length"])
    model = CustomEncoder(dict_size, config["model"]["d_model"], config["model"]["num_layers"], config["model"]["num_heads"], config["model"]["d_ff"], config["model"]["max_seq_length"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    for epoch in range(1, config["training"]["num_epochs"]+1):
      model.train()
      total_loss = 0
      for batch in train_loader:

        optimizer.zero_grad()

        query_tkns = batch["query_input_ids"]
        query_masks = batch["query_attention_mask"]
        passage_tkns = batch["passage_input_ids"]
        passage_masks = batch["passage_attention_mask"]

        query_embeds = model(query_tkns, query_masks)
        passage_embeds = model(passage_tkns, passage_masks)

        loss = contrastive_loss(query_embeds, passage_embeds, config["training"]["temperature"])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
      
      avg_loss = total_loss / len(train_loader)
      print(f"Epoch {epoch}: loss = {avg_loss}")
    
      save_checkpoint(model, optimizer, epoch, config["training"]["checkpoint_dir"])


def evaluate_step(model, val_loader, device):
    """Optional: run a validation pass and return average loss / retrieval metrics.

    TODO: implement if you want validation during training (recommended).
    """
    raise NotImplementedError
