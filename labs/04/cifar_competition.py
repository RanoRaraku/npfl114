#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from cifar10_torch import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.epoch = 0
        self.in_channels=3
        self.in_height=32
        self.in_width=32
        self.num_classes=10
        self.dense_dim = 256
        self.init_filters = 32

        self.vgg1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.init_filters, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.init_filters),
            nn.Conv2d(self.init_filters, self.init_filters, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.init_filters),
            nn.MaxPool2d((2,2)),
        )
        self.vgg2 = nn.Sequential(
            nn.Conv2d(self.init_filters, self.init_filters * 2, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.init_filters * 2),
            nn.Conv2d(self.init_filters * 2, self.init_filters * 2, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.init_filters * 2),
            nn.MaxPool2d((2,2)),
        )
        self.vgg3 = nn.Sequential(
            nn.Conv2d(self.init_filters * 2, self.init_filters * 4, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.init_filters * 4),
            nn.Conv2d(self.init_filters * 4, self.init_filters * 4, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(self.init_filters * 4),
            nn.MaxPool2d((2,2)),
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout2d(0.1)
        self.dense1 = nn.Linear(2048, self.dense_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.dense_dim, self.num_classes)

    def forward(self, inputs):

        x = self.vgg1(inputs)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.out(x)

        return x

    def eval_accuracy(self, dloader):

        self.eval()
        total_corr = 0
        num_iters = len(dloader)
        for samples, labels in dloader:
            y_hat = self(samples)
            y_pred = torch.argmax(y_hat, dim=-1)
            total_corr +=  torch.sum(y_pred == torch.argmax(labels, dim=-1))
        print(f"dev_accuracy={total_corr/(num_iters * 32) * 100:.2f} %")

    def train_epoch(self, train_dloader, loss_fn, optim):
        
        self.train()
        for idx, (samples, labels) in enumerate(train_dloader):

            # Run inference
            y_hat = self(samples)
            loss = loss_fn(y_hat, labels)

            # Update params
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.epoch += 1

class CosineLRWithWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr=0.001, final_lr=0.0001, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super(CosineLRWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr + (self.final_lr - self.base_lr) * self.last_epoch / self.warmup_epochs for _ in self.optimizer.param_groups]

        cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
        return [self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (1 + cosine_decay) for _ in self.optimizer.param_groups]

def main(args: argparse.Namespace) -> None:
    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load train data
    cifar = CIFAR10()
    train_dloader = DataLoader(cifar.train, args.batch_size, shuffle=True)
    dev_dloader = DataLoader(cifar.dev, args.batch_size, shuffle=False)
    test_dloader = DataLoader(cifar.test, args.batch_size, shuffle=False)

    # TODO: Create the model and train it
    model = VGG()
    optim = torch.optim.AdamW(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    scheduler = CosineLRWithWarmup(optim, 2, args.epochs)

    for _ in range(10):
        model.train_epoch(train_dloader, loss_fn, optim)
        model.eval_accuracy(dev_dloader)
        scheduler.step()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
