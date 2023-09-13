#!/usr/bin/env python3
import argparse
import datetime
import os
import re

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset_torch import MorphoDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use")


class Model(nn.Module):

    def __init__(self, vocab_size, num_classes):
        super(Model, self).__init__()
        self.epoch = 0

        self.input = nn.Identity()
        self.form_embedd = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        self.linear = nn.Linear(256, num_classes)

    def forward(self, inputs):
        x = self.input(inputs)
        x = self.form_embedd(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
        
    def eval_accuracy(self, dloader):

        self.eval()
        total, total_corr = 0, 0
        for batch in dloader:
            seq_lens = batch["sequence_lengths"]
            max_seq_len = torch.max(seq_lens)
            mask = torch.arange(max_seq_len).expand(len(seq_lens), max_seq_len) < seq_lens.unsqueeze(1)

            y_hat = self(batch["forms"])
            total_corr += torch.sum(torch.argmax(y_hat[mask], dim=-1) == torch.argmax(batch["tags"], dim=-1))
            total += torch.sum(seq_lens)
        print(f"dev_accuracy={total_corr/total * 100:.2f} %")

    def train_epoch(self, train_dloader, loss_fn, optim):
        
        self.train()
        total_loss = 0
        for batch in train_dloader:
            seq_lens = batch["sequence_lengths"]
            max_seq_len = torch.max(seq_lens)
            mask = torch.arange(max_seq_len).expand(len(seq_lens), max_seq_len) < seq_lens.unsqueeze(1)

            # Run inference
            y_hat = self(batch["forms"])
            loss = loss_fn(y_hat[mask], batch["tags"][mask])

            # Update params
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        self.epoch += 1

def main(args: argparse.Namespace) -> None:


    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")

    # TODO: Create the model and train it
    model = Model(morpho.train.unique_forms, morpho.train.unique_tags)
    optim = torch.optim.AdamW(model.parameters())
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


    # analyses = MorphoAnalyzer("czech_pdt_analyses")

    train_dloader = DataLoader(
        morpho.train,
        args.batch_size,
        collate_fn=morpho.collate,
        shuffle=True,
    )
 
    dev_dloader = DataLoader(
        morpho.dev,
        args.batch_size,
        collate_fn=morpho.collate,
    )


    for _ in range(1):
        model.train_epoch(train_dloader, loss_fn, optim)
        model.eval_accuracy(train_dloader)
        #scheduler.step()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
