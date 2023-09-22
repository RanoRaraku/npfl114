import wandb
import datetime
import time
from typing import Optional, Any
import torch
import torch.nn as nn

from Morpho import MorphoDataset
from model import SimpleRNN, Seq2Seq


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
        corr += torch.sum(
            torch.argmax(y_hat[mask], dim=-1) == torch.argmax(tags[mask], dim=-1)
        )
        total_samples += torch.sum(words_num)

        if loss_fn:
            loss = loss_fn(y_hat[mask], tags[mask])
            total_loss += loss.item()

    return corr / total_samples, total_loss / len(dloader)

def train_epoch(model, optim, loss_fn, train_dataloader, dev_dataloader, logger):

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
        y_hat = model(words, words_num, chars)
        loss = loss_fn(y_hat[mask], tags[mask])

        # Update params
        optim.zero_grad()
        loss.backward()
        optim.step()

        if logger:
            logger.log({"train_loss": loss.item()})

        exit()
    model.epoch += 1

    # log metrics to wandb
    dev_acc, dev_loss = eval_accuracy(dev_dataloader, loss_fn)
    end_time = time.time()
    if logger:
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


morpho = MorphoDataset("czech_pdt")

simple_rnn_args = {
    "batch_size":128,
    "epochs":20,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "dataset":"czech_pdt",
    "model":"SimpleRNN",
    "we_dim":128,
    "hidden_size":1024,
    "num_layers":4,
    "dropout":0.1,
    "word_vocab_size": morpho.train.unique_forms,
    "char_vocab_size": morpho.train.unique_chars,
    "num_classes":morpho.train.unique_tags,
    "label_smoothing":0.1,
    "packed_sequences": True,
    "characters": True,
}
seq2seq_args = {
    "batch_size":2,
    "epochs":1,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
    "dataset":"czech_pdt",
    "model":"Seq2Seq",
    "we_dim":128,
    "hidden_size":128,
    "num_layers":1,
    "dropout":0.1,
    "word_vocab_size": morpho.train.unique_forms,
    "char_vocab_size": morpho.train.unique_chars,
    "num_classes":morpho.train.unique_tags,
    "label_smoothing":0.1,
    "packed_sequences": True,
    "characters": True,
}



# 1) SimpleRNN
#model = SimpleRNN(simple_rnn_args).to(simple_rnn_args["device"])
# 2) Seq2Seq
model = Seq2Seq(seq2seq_args).to(seq2seq_args["device"])


optim = torch.optim.AdamW(model.parameters())
loss_fn = nn.CrossEntropyLoss(label_smoothing=seq2seq_args["label_smoothing"])
train_dloader = morpho.train.to_dataloader(seq2seq_args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dataloader(seq2seq_args["batch_size"], shuffle=False)


# wandb.login()
# run_name =  datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# wandb.init(project="tagger_competition", name=run_name, config=simple_rnn_args)
# for _ in range(simple_rnn_args["epochs"]):
#     train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, wandb)
# wandb.finish()


for _ in range(seq2seq_args["epochs"]):
    train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, wandb)

