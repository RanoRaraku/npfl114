import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from typing import Optional, Any

class SimpleRNN(nn.Module):

    # zvysit dropout
    # pridat chars
    def __init__(self, args):
        super(SimpleRNN, self).__init__()
        self.epoch = 0
        self.args = args
        self.device = args["device"]

        self.word_embedd = nn.Embedding(args["word_vocab_size"], args["we_dim"])
        self.word_lstm = nn.LSTM(
            input_size=args["we_dim"],
            hidden_size=args["hidden_size"],
            num_layers=args["num_layers"],
            batch_first=True,
            dropout=args["dropout"],
            bidirectional=True
        )
        self.linear = nn.Linear(args["hidden_size"] * 2, args["num_classes"])

    def forward(self, input, seq_lens):
        x = self.word_embedd(input)
        x = pack_padded_sequence(x, seq_lens.to("cpu"), batch_first=True, enforce_sorted=False)
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.linear(x)
        return x
    
    def eval_accuracy(self, dloader, loss_fn:Optional[Any] = None):
        """
        Returns accuracy and optionally per_sample loss.
        """
        self.eval()
        total_loss, total_samples, corr = 0, 0, 0
        for batch in dloader:
            seq_lens = batch["sequence_lens"].to(self.device)
            words = batch["words"].to(self.device)
            tags = batch["tags"].to(self.device)
            max_seq_len = torch.max(seq_lens)
            mask = torch.arange(max_seq_len, device=self.device).expand(len(seq_lens), max_seq_len) < seq_lens.unsqueeze(1)

            y_hat = self(words, seq_lens)
            corr += torch.sum(torch.argmax(y_hat[mask], dim=-1) == torch.argmax(tags[mask], dim=-1))
            total_samples += torch.sum(seq_lens)

            if loss_fn:
                loss = loss_fn(y_hat[mask],tags[mask])
                total_loss += loss.item()


        return corr/total_samples, total_loss/total_samples

    def train_epoch(self, train_dloader, dev_dloader, loss_fn, optim, logger:Optional[Any] = None):

        start_time = time.time()
        self.train()
        for batch in train_dloader:
            seq_lens = batch["sequence_lens"].to(self.device)
            words = batch["words"].to(self.device)
            tags = batch["tags"].to(self.device)
            max_seq_len = torch.max(seq_lens)
            mask = torch.arange(max_seq_len, device=self.device).expand(len(seq_lens), max_seq_len) < seq_lens.unsqueeze(1)

            # Run inference
            y_hat = self(words, seq_lens)
            loss = loss_fn(y_hat[mask],tags[mask])

            # Update params
            optim.zero_grad()
            loss.backward()
            optim.step()

            if logger:
                logger.log({"train_loss":loss.item()})

        self.epoch += 1

        dev_acc, dev_loss = self.eval_accuracy(dev_dloader, loss_fn)
        end_time = time.time()

        # log metrics to wandb
        if logger:
            logger.log(
                {
                    "epoch_time":end_time-start_time,
                    "dev_loss":dev_loss,
                    "dev_acc":dev_acc
                }
            )
