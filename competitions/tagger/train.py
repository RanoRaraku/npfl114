import torch.nn as nn
import wandb

import torch

from model import Seq2Seq, SimpleRNN, train_epoch
from Morpho import MorphoDataset
import datetime


morpho = MorphoDataset("czech_pdt", add_sos_eos=True)

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
# seq2seq_args = {
#     "batch_size": 64,
#     "epochs": 10,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "dataset": "czech_pdt",
#     "model": "Seq2Seq",
#     "we_dim": 256,
#     "hidden_size": 1024,
#     "num_layers": 3,
#     "dropout": 0.1,
#     "word_vocab_size": morpho.train.unique_forms,
#     "char_vocab_size": morpho.train.unique_chars,
#     "num_classes": morpho.train.unique_tags,
#     "label_smoothing": 0.1,
#     "packed_sequences": True,
#     "characters": True,
# }

# args = simple_rnn_args
# model = SimpleRNN(args).to(args["device"])
# #model = Seq2Seq(args).to(args["device"])


# optim = torch.optim.AdamW(model.parameters())
# loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
# train_dloader = morpho.train.to_dataloader(args["batch_size"], shuffle=True)
# dev_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)

# for _ in range(seq2seq_args["epochs"]):
#     train_epoch(model, train_dloader, dev_dloader, loss_fn, optim)
