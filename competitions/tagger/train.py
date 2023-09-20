import wandb
import datetime

import torch
import torch.nn as nn

from Morpho import MorphoDataset
from model import SimpleRNN

wandb.login()

morpho = MorphoDataset("czech_pdt")

simple_rnn_args = {
    "batch_size":128,
    "epochs":20,
    "device":"cpu",
    "dataset":"czech_pdt",
    "model":"SimpleRNN",
    "we_dim":256,
    "hidden_size":1024,
    "num_layers":4,
    "dropout":0.1,
    "word_vocab_size": morpho.train.unique_forms,
    "num_classes":morpho.train.unique_tags,
    "label_smoothing":0.1,
    "packed_sequences": True,
}
model = SimpleRNN(simple_rnn_args).to(simple_rnn_args["device"])
optim = torch.optim.AdamW(model.parameters())
loss_fn = nn.CrossEntropyLoss(label_smoothing=simple_rnn_args["label_smoothing"])
train_dloader = morpho.train.to_dloader(simple_rnn_args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dloader(simple_rnn_args["batch_size"], shuffle=False)


run_name =  datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(project="tagger_competition", name=run_name, config=simple_rnn_args)
for _ in range(simple_rnn_args["epochs"]):
    model.train_epoch(train_dloader, dev_dloader, loss_fn, optim, wandb)
wandb.finish()

