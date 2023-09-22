import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
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
            bidirectional=True,
        )
        self.char_embedd = nn.Embedding(args["char_vocab_size"], args["we_dim"])
        self.char_lstm = nn.LSTM(
            input_size=args["we_dim"],
            hidden_size=2*args["hidden_size"],
            num_layers=1,
            batch_first=True,
            dropout=args["dropout"],
            bidirectional=False,
        )
        self.linear = nn.Linear(args["hidden_size"] * 2, args["num_classes"])

    def forward(self, words, words_num, chars):
        # word LSTM
        x = self.word_embedd(words)
        x = pack_padded_sequence(
            x, words_num.to("cpu"), batch_first=True, enforce_sorted=False
        )
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Character LSTM
        # embedd a word at a time where word is a sequence of characters
        y = [[self.char_embedd(word) for word in sent] for sent in chars]
        # pack words in sentence into one packed sequence
        y = [pack_sequence(embeddings, enforce_sorted=False) for embeddings in y]
        # output hidden state h_n for last character in a word
        y = [self.char_lstm(packed)[1][0].squeeze(0) for packed in y]
        # pad to match dimensionality of word_lstm
        y = pad_sequence(y, batch_first=True)

        z = self.linear(x + y)
        return z

    def eval_accuracy(self, dloader, loss_fn: Optional[Any] = None):
        """
        Returns accuracy and optionally per_sample loss.
        """
        self.eval()
        total_loss, total_samples, corr = 0, 0, 0
        for batch in dloader:
            words_num = batch["words_num"].to(self.device)
            words = batch["words"].to(self.device)
            chars = batch["chars"]
            tags = batch["tags"].to(self.device)
            
            max_seq_len = torch.max(words_num)
            mask = torch.arange(max_seq_len, device=self.device).expand(
                len(words_num), max_seq_len
            ) < words_num.unsqueeze(1)
            y_hat = self(words, words_num, chars)
            corr += torch.sum(
                torch.argmax(y_hat[mask], dim=-1) == torch.argmax(tags[mask], dim=-1)
            )
            total_samples += torch.sum(words_num)

            if loss_fn:
                loss = loss_fn(y_hat[mask], tags[mask])
                total_loss += loss.item()

        return corr / total_samples, total_loss / len(dloader)

    def train_epoch(
        self, train_dloader, dev_dloader, loss_fn, optim, logger: Optional[Any] = None
    ):
        start_time = time.time()
        self.train()
        for batch in train_dloader:
            words_num = batch["words_num"].to(self.device)
            words = batch["words"].to(self.device)
            chars = batch["chars"]
            tags = batch["tags"].to(self.device)

            max_words_num = torch.max(words_num)
            mask = torch.arange(max_words_num, device=self.device).expand(
                len(words_num), max_words_num
            ) < words_num.unsqueeze(1)

            # Run inference
            y_hat = self(words, words_num, chars)
            loss = loss_fn(y_hat[mask], tags[mask])

            # Update params
            optim.zero_grad()
            loss.backward()
            optim.step()

            if logger:
                logger.log({"train_loss": loss.item()})
        self.epoch += 1

        # log metrics to wandb
        dev_acc, dev_loss = self.eval_accuracy(dev_dloader, loss_fn)
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
                "epoch": self.epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": loss.item(),
            },
            f"{self.epoch}.pt",
        )
