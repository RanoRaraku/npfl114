import datetime

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR

from Morpho import MorphoDataset
from transformer import (Transformer, eval_accuracy, train_epoch,
                         train_transformer)

morpho = MorphoDataset("czech_pdt")

args = {
    "batch_size": 2,
    "epochs": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "transformer",
    "heads": 1,
    "input_dropout": 0.2,
    "model_dim": 512,
    "keys_dim": 64,
    "values_dim": 64,
    "max_seq_len": morpho.max_length,
    "encoder_stack_size": 2,
    "decoder_stack_size": 2,
    "input_vocab_size": morpho.train.unique_words,
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

# wandb.login()
# run_name =  datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# wandb.init(project="tagger_competition", name=run_name, config=args)
for epoch in range(args["epochs"]):
    train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, scheduler, None)
    # eval_accuracy(model, dev_dloader, loss_fn)
    # train_epoch(train_dloader, loss_fn, optim, args)
# wandb.finish()
