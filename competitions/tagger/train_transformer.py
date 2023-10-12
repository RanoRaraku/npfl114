import datetime

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR

from Morpho import MorphoDataset
from transformer import Transformer, train_epoch

morpho = MorphoDataset("czech_pdt")

args = {
    "batch_size": 2,
    "epochs": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "Seq2Seq",
    "model_dim": 512,
    "max_seq_len": morpho.max_length,
    "encoder_stack_size": 2,
    "decoder_stack_size": 2,
    "word_vocab_size": morpho.train.unique_forms,
    "num_classes": morpho.train.unique_tags,
    "label_smoothing": 0.1,
    "packed_sequences": False,
    "characters": False,
}

model = Transformer(args)
optim = torch.optim.AdamW(model.parameters())
loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
train_dloader = morpho.train.to_dataloader(args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)
scheduler = StepLR(optim, step_size=1, gamma=0.5)


wandb = None
for epoch in range(args["epochs"]):
    train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, scheduler, wandb)
