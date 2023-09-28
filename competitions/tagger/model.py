import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)
import pathlib
import time
from typing import Any, Optional


class SimpleRNN(nn.Module):
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
            hidden_size=2 * args["hidden_size"],
            num_layers=1,
            batch_first=True,
            dropout=args["dropout"],
            bidirectional=False,
        )
        self.linear = nn.Linear(args["hidden_size"] * 2, args["num_classes"])

    def forward(self, words, words_num, chars, targets=None):
        # word LSTM
        x = self.word_embedd(words)
        x = pack_padded_sequence(
            x, words_num.to("cpu"), batch_first=True, enforce_sorted=False
        )
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        # Character LSTM
        # embedd a word at a time where word is a sequence of characters
        y = [
            [self.char_embedd(word.to(self.device)) for word in sent] for sent in chars
        ]
        # pack words in sentence into one packed sequence
        y = [pack_sequence(embeddings, enforce_sorted=False) for embeddings in y]
        # output hidden state h_n for last character in a word
        y = [self.char_lstm(packed)[1][0].squeeze(0) for packed in y]
        # pad to match dimensionality of word_lstm
        y = pad_sequence(y, batch_first=True)

        z = self.linear(x + y)

        return z


class Seq2Seq(nn.Module):
    """
    Encoder-Decoder model. Encoder is LSTM, consumes words and chars
    and outputs a context vector. Decoder is auto-regressive where tag_t
    is used as input to predict tag_t+1.
    """

    class Encoder(nn.Module):
        def __init__(self, args):
            super(Seq2Seq.Encoder, self).__init__()
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
                hidden_size=2 * args["hidden_size"],
                num_layers=1,
                batch_first=True,
                dropout=args["dropout"],
                bidirectional=False,
            )

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
            y = [
                [self.char_embedd(word.to(self.device)) for word in sent]
                for sent in chars
            ]
            # pack words in sentence into one packed sequence
            y = [pack_sequence(embeddings, enforce_sorted=False) for embeddings in y]
            # output hidden state h_n for last character in a word
            y = [self.char_lstm(packed)[1][0].squeeze(0) for packed in y]
            # pad to match dimensionality of word_lstm
            y = pad_sequence(y, batch_first=True)

            z = x + y

            # return last/first item in forward/backward sequence
            # tensor is of shape (B, 1, 2*hidden_dim)
            return (
                torch.cat((z[:, -1, :128], z[:, 0, 128:]), -1)
                .unsqueeze(1)
                .permute(1, 0, 2)
            )

    class Decoder(nn.Module):
        def __init__(self, args):
            super(Seq2Seq.Decoder, self).__init__()
            self.epoch = 0
            self.args = args
            self.device = args["device"]

            self.embedd = nn.Embedding(args["word_vocab_size"], args["we_dim"])
            self.gru = nn.GRU(
                input_size=args["we_dim"],
                hidden_size=2 * args["hidden_size"],
                num_layers=1,
                batch_first=True,
                dropout=args["dropout"],
                bidirectional=False,
            )
            self.out = nn.Linear(2 * args["hidden_size"], args["num_classes"])

        def forward_step(self, hidden, input):
            """
            1 krok RNN, 1 item sekvencie.
            """
            x = self.embedd(input)
            x, h = self.gru(x, hidden)
            x = self.out(x)
            return x, h

        def forward(self, hidden, inputs_num, targets=None):
            """
            Word->Tag is 1:1 mapping, use this info to setup output size.
            tag_length == word_length + 2 and <BOS> is 0 and <EOS> is 1.
            """
            batch_size = hidden.size(1)
            max_len = torch.max(inputs_num).item()
            inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            outputs = []

            for i in range(max_len):
                output, hidden = self.forward_step(hidden, inputs)
                outputs.append(output)

                # print(f"targets:{targets.shape}")
                if targets is not None:
                    # Teacher forcing: Feed the target as the next input
                    inputs = targets[:, i].unsqueeze(1)  # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    _, topi = output.topk(1)
                    inputs = topi.squeeze(-1).detach()  # detach from history as input

            outputs = torch.cat(outputs, dim=1)
            return outputs

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.epoch = 0
        self.args = args
        self.device = args["device"]

        self.encoder = Seq2Seq.Encoder(args)
        self.decoder = Seq2Seq.Decoder(args)

    def forward(self, words, words_num, chars, targets=None):
        encoded = self.encoder(words, words_num, chars)
        decoded = self.decoder(encoded, words_num, targets)

        return decoded


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