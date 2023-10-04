import datetime

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR

from model import Seq2Seq, Seq2SeqAtt, SimpleRNN, train_epoch
from Morpho import MorphoDataset

morpho = MorphoDataset("czech_pdt")

simple_rnn_args = {
    "batch_size": 2,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "SimpleRNN",
    "we_dim": 64,
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.1,
    "word_vocab_size": morpho.train.unique_forms,
    "char_vocab_size": morpho.train.unique_chars,
    "num_classes": morpho.train.unique_tags,
    "label_smoothing": 0.1,
    "packed_sequences": True,
    "characters": True,
}
seq2seq_args = {
    "batch_size": 2,
    "epochs": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "Seq2Seq",
    "we_dim": 32,
    "encoder_hidden_size": 64,
    "attention_size": 48,
    "word_encoder_layers": 1,
    "char_encoder_layers": 1,
    "dropout": 0.1,
    "word_vocab_size": morpho.train.unique_forms,
    "char_vocab_size": morpho.train.unique_chars,
    "num_classes": morpho.train.unique_tags,
    "label_smoothing": 0.1,
    "packed_sequences": True,
    "characters": True,
}

args = seq2seq_args
# model = SimpleRNN(args).to(args["device"])
# model = Seq2Seq(args).to(args["device"])
model = Seq2SeqAtt(args).to(args["device"])


optim = torch.optim.AdamW(model.parameters())
loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
train_dloader = morpho.train.to_dataloader(args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)
scheduler = StepLR(optim, step_size=1, gamma=0.5)

# wandb.login()
# run_name =  f"debug-{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
# wandb.init(project="tagger_competition", name=run_name, config=args)
wandb = None
for _ in range(seq2seq_args["epochs"]):
    train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, scheduler, wandb)

# wandb.finish()
