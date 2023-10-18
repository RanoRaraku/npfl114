import datetime

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR

from Morpho import MorphoDataset
from transformer import Transformer, eval_accuracy, train_epoch

morpho = MorphoDataset("czech_pdt")

args = {
    "batch_size": 128,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "transformer",
    "heads": 8,
    "input_dropout": 0,
    "model_dim": 512,
    "keys_dim": 512,
    "values_dim": 512,
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
optim = torch.optim.AdamW(model.parameters(), 0.00075)
loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
train_dloader = morpho.train.to_dataloader(args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)
scheduler = StepLR(optim, step_size=1, gamma=0.75)

wandb.login()
run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(project="tagger_competition", name=run_name, config=args)
for epoch in range(args["epochs"]):
    train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, scheduler, None)
wandb.finish()
