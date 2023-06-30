#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Model(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.epoch = 0
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 10, 3, 2, "valid"),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, 2, "valid"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(720, 200),
            nn.ReLU(),
        )
        self.dense2 = nn.Linear(400, 200)
        self.relu = nn.ReLU()
        self.dense3 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()
        self.dense4 = nn.Linear(200, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Expect x to be batch of images which we split into two
        """
        
        # shared Conv2D stack
        x_0 = self.conv_stack(x[0])
        x_1 = self.conv_stack(x[1])

        # Direct comparison
        direct_prediction = torch.cat((x_0, x_1), dim=1)
        direct_prediction = self.dense2(direct_prediction)
        direct_prediction = self.relu(direct_prediction)
        direct_prediction = self.dense3(direct_prediction)
        direct_prediction = self.sigmoid(direct_prediction)

        # Digit prediction
        digit_0 = self.softmax(self.dense4(x_0))
        digit_1 = self.softmax(self.dense4(x_1))

        return {
            "direct_prediction": direct_prediction,
            "digit_0": digit_0,
            "digit_1": digit_1,
        }

def create_dloader(train:bool):
    dset = torchvision.datasets.MNIST(
        root="./data",
        train=train,
        transform=transforms.ToTensor(),
        download=True,
    )
    return dset


def evaluate(model, test_dloader):
    total, corr_indirect, corr_direct, corr_digit_0, corr_digit_1 = 0, 0, 0, 0, 0
    for (samples, targets) in test_dloader:

        # Create tuples (samples_0, samples_1) && (targets_0, targets_1)
        samples = (samples[:25,:,:,:], samples[25:,:,:,:])
        targets = (targets[:25], targets[25:])
        targets_direct = (targets[0] > targets[1]).to(torch.float32).unsqueeze(-1)

        # Run inference
        out = model(samples)

        total += out["digit_0"].shape[0]
        corr_digit_0 += (torch.argmax(out["digit_0"],1) == targets[0]).sum()
        corr_digit_1 += (torch.argmax(out["digit_1"],1) == targets[1]).sum()
        corr_direct += ((out["direct_prediction"] > 0.5) == targets_direct).sum()
        indirect_prediction = (
            torch.argmax(out["digit_0"], dim=1) > torch.argmax(out["digit_1"], dim=1)
        ).to(torch.float32).unsqueeze(-1)
        corr_indirect += (indirect_prediction == targets_direct).sum()

    acc_direct = 100 * corr_direct / total
    acc_indirect = 100 * corr_indirect / total
    acc_digit_0 = 100 * corr_digit_0 / total
    acc_digit_1 = 100 * corr_digit_1 / total

    return {
        "direct": acc_direct,
        "digit_0": acc_digit_0,
        "digit_1": acc_digit_1,
        "indirect": acc_indirect,
    }


def train_epoch(model, train_dloader, loss_fn_direct, loss_fn_pred, optim):

    model.epoch += 1
    model.train()
    num_iters = len(train_dloader)
    for idx, (samples, targets) in enumerate(train_dloader):

        # Create tuples (samples_0, samples_1) && (targets_0, targets_1)
        samples = (samples[:25,:,:,:],samples[25:,:,:,:])
        targets = (targets[:25], targets[25:])
        targets_direct = (targets[0] > targets[1]).to(torch.float32).unsqueeze(-1)

        # Run inference
        out = model(samples)

        # Calculate all losses
        loss_direct = loss_fn_direct(out['direct_prediction'], targets_direct)
        loss_pred_1 = loss_fn_pred(out['digit_0'], targets[0])
        loss_pred_2 = loss_fn_pred(out['digit_1'], targets[1])
        loss_pred = torch.mean(torch.cat((loss_pred_1,loss_pred_2)))
        total_loss = loss_direct + loss_pred

        # Update params
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if (idx + 1) % 100 == 0:
            print(
                f"iter={idx+1}/{num_iters}, "
                f"direct_loss={loss_direct.item():.4f}, "
                f"loss_pred_1={torch.mean(loss_pred_1).item():.4f}, "
                f"loss_pred_2={torch.mean(loss_pred_2).item():.4f}, "
                f"total_loss={total_loss.item():.4f}."
            )

def main(args: argparse.Namespace) -> Dict[str, float]:

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Create the model
    model = Model()
    optim = torch.optim.AdamW(model.parameters())
    loss_fn_direct = nn.BCELoss()
    loss_fn_pred = nn.NLLLoss(reduction='none')

    # Construct suitable datasets from the MNIST data.
    train_dset = create_dloader(train=True)
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_dset = create_dloader(train=False)
    test_dloader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Train
    for epoch in range(args.epochs):
        train_epoch(model, train_dloader, loss_fn_direct, loss_fn_pred, optim)

        results = evaluate(model, test_dloader)
        print(
            f"Test Data: epoch={model.epoch}, "
            f"Accuracy direct: {results['direct']:.2f} [%], "
            f"Accuracy digit_0: {results['digit_0']:.2f} [%], "
            f"Accuracy digit_1: {results['digit_1']:.2f} [%], "
            f"Accuracy indirect: {results['indirect']:.2f} [%]."
        )

        

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)