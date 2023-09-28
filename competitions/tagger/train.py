import pathlib
import time
from typing import Any, Optional

import torch


def eval_accuracy(model, dloader, loss_fn: Optional[Any] = None):
    """
    Returns accuracy and optionally per_sample loss.
    """
    model.eval()
    total_loss, total_samples, corr = 0, 0, 0
    for batch in dloader:
        words_num = batch["words_num"].to(model.device)
        words = batch["words"].to(model.device)
        chars = batch["chars"]
        tags = batch["tags"].to(model.device)

        max_seq_len = torch.max(words_num)
        mask = torch.arange(max_seq_len, device=model.device).expand(
            len(words_num), max_seq_len
        ) < words_num.unsqueeze(1)
        y_hat = model(words, words_num, chars)
        corr += torch.sum(torch.argmax(y_hat[mask], dim=-1) == tags[mask])
        total_samples += torch.sum(words_num)

        if loss_fn:
            loss = loss_fn(y_hat[mask], tags[mask])
            total_loss += loss.item()

    return corr / total_samples, total_loss / len(dloader)


def load_checkpoint(path, model, optim):
    last_checkpoint = 0
    for f in pathlib.Path(".").iterdir():
        if ".pt" in str(f) and int(f.stem) > last_checkpoint:
            path = str(f)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    model.epoch = epoch
    return model, optim, loss


def train_epoch(
    model,
    train_dataloader,
    dev_dataloader,
    loss_fn,
    optim,
    logger: Optional[Any] = None,
):
    start_time = time.time()
    model.train()
    for batch in train_dataloader:
        words_num = batch["words_num"].to(model.device)
        words = batch["words"].to(model.device)
        chars = batch["chars"]
        tags = batch["tags"].to(model.device)

        max_words_num = torch.max(words_num)
        mask = torch.arange(max_words_num, device=model.device).expand(
            len(words_num), max_words_num
        ) < words_num.unsqueeze(1)

        # Run inference
        y_hat = model(words, words_num, chars, tags)
        loss = loss_fn(y_hat[mask], tags[mask])

        # Update params
        optim.zero_grad()
        loss.backward()
        optim.step()

        if logger is not None:
            logger.log({"train_loss": loss.item()})

    model.epoch += 1

    # log metrics to wandb
    dev_acc, dev_loss = eval_accuracy(model, dev_dataloader, loss_fn)
    end_time = time.time()
    if logger is not None:
        logger.log(
            {
                "epoch_time": end_time - start_time,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }
        )

    # save checkpoint
    torch.save(
        {
            "epoch": model.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "loss": loss.item(),
        },
        f"tagger.{model.epoch}.pt",
    )
