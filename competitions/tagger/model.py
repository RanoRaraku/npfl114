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
        """
        TN: character encoded pricitam ku kazdemu slovu ale ked to vraciam ako
        vystup encoderu tak mi chybaju pre ine prve a posledne slovo
        TN: pridat <EOS> na koniec a spravne matchovat pri loss calc
        """

        def __init__(self, args):
            super(Seq2Seq.Encoder, self).__init__()
            self.args = args
            self.device = args["device"]

            self.word_embedd = nn.Embedding(args["word_vocab_size"], args["we_dim"])
            self.word_lstm = nn.LSTM(
                input_size=args["we_dim"],
                hidden_size=args["encoder_hidden_size"],
                num_layers=args["word_encoder_layers"],
                batch_first=True,
                dropout=args["dropout"],
                bidirectional=True,
            )
            self.char_embedd = nn.Embedding(args["char_vocab_size"], args["we_dim"])
            self.char_lstm = nn.LSTM(
                input_size=args["we_dim"],
                hidden_size=args["encoder_hidden_size"],
                num_layers=args["char_encoder_layers"],
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

            # return last/first item in forward/backward sequence
            # tensor is of shape (B, 1, 2*hidden_dim)
            return (
                torch.cat((x[:, -1, :int(x.shape[-1]/2)], x[:, 0, int(x.shape[-1]/2):]), -1)
                .unsqueeze(1)
                .permute(1, 0, 2),
                y, 
            )

    class Decoder(nn.Module):
        def __init__(self, args):
            super(Seq2Seq.Decoder, self).__init__()
            self.args = args
            self.device = args["device"]

            self.embedding = nn.Embedding(args["word_vocab_size"], args["we_dim"])
            self.gru = nn.GRU(
                input_size=args["we_dim"] + args["encoder_hidden_size"],
                hidden_size=2 * args["decoder_hidden_size"],
                num_layers=1,
                batch_first=True,
                dropout=args["dropout"],
                bidirectional=False,
            )
            self.out = nn.Linear(2 * args["decoder_hidden_size"], args["num_classes"])

        def forward_step(self, hidden, input, chars_encoded):
            """
            1 krok RNN, 1 item sekvencie.
            """
            x = self.embedding(input)
            x, h = self.gru(torch.cat((x, chars_encoded.unsqueeze(1)), -1), hidden)
            x = self.out(x)
            return x, h

        def forward(self, context, chars_encoded, words_num, targets=None):
            """
            Word->Tag is 1:1 mapping, use this info to setup output size.
            """
            hidden = context
            batch_size = hidden.size(1)
            max_len = torch.max(words_num).item()
            inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            outputs = []

            for i in range(max_len):
                output, hidden = self.forward_step(hidden, inputs, chars_encoded[:,i,:])
                outputs.append(output)

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
        context, chars_encoded = self.encoder(words, words_num, chars)
        decoded = self.decoder(context, chars_encoded, words_num, targets)

        return decoded


class Seq2SeqAttn(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, args):
            super(Seq2SeqAttn.Encoder, self).__init__()
            self.device = args["device"]

            self.word_embedd = nn.Embedding(args["word_vocab_size"], args["we_dim"])
            self.word_lstm = nn.LSTM(
                input_size=args["we_dim"],
                hidden_size=args["encoder_hidden_size"],
                num_layers=args["word_encoder_layers"],
                batch_first=True,
                dropout=args["dropout"],
                bidirectional=True,
            )
            self.char_embedd = nn.Embedding(args["char_vocab_size"], args["we_dim"])
            self.char_lstm = nn.LSTM(
                input_size=args["we_dim"],
                hidden_size=2 * args["encoder_hidden_size"],
                num_layers=args["char_encoder_layers"],
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

            # tensor is of shape (B, T, 2*hidden_dim)
            return z

    class BahdanauAttention(nn.Module):
        def __init__(self, args):
            super(Seq2SeqAttn.BahdanauAttention, self).__init__()
            self.U = nn.Linear(2 * args["encoder_hidden_size"], args["attention_size"])
            self.W = nn.Linear(2 * args["encoder_hidden_size"], args["attention_size"])
            self.V = nn.Linear(args["attention_size"], 1, bias=False)

        def forward(self, encoder_outputs, decoder_hidden):
            e = self.V(torch.tanh(self.U(encoder_outputs) + self.W(decoder_hidden)))
            alpha = nn.functional.softmax(e, -1)
            c = torch.bmm(alpha, encoder_outputs)
            return c

    class Decoder(nn.Module):
        """
        1) Inicializovat dekoder poslednym stavom enkoderu
        """

        def __init__(self, args):
            super(Seq2SeqAttn.Decoder, self).__init__()
            self.device = args["device"]

            self.embedding = nn.Embedding(args["word_vocab_size"], args["we_dim"])
            self.attention = Seq2SeqAttn.BahdanauAttention(args)
            self.gru = nn.GRU(
                input_size=args["we_dim"],
                hidden_size=args["decoder_hidden_size"],
                num_layers=1,
                batch_first=True,
                dropout=args["dropout"],
                bidirectional=False,
            )
            self.out = nn.Linear(args["decoder_hidden_size"], args["num_classes"])

        def forward_step(self, encoder_outputs, decoder_hidden, decoder_inputs):
            """ """
            x = self.embedding(decoder_inputs)
            c = self.attention(encoder_outputs, decoder_hidden)
            x, h = self.gru(torch.cat((x, c), dim=-1), decoder_hidden.permute(1, 0, 2))
            x = self.out(x)
            return x, h

        def forward(self, encoder_outputs, inputs_num, targets=None):
            """

            1) Init decoder h_0 to encoder output
            2) Word->Tag is 1:1 mapping, use this info to setup max_len
            3)
            """
            decoder_hidden = encoder_outputs
            batch_size = encoder_outputs.size(0)
            max_len = torch.max(inputs_num).item()
            decoder_inputs = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )
            decoder_outputs = []

            for i in range(max_len):
                output, decoder_hidden = self.forward_step(
                    encoder_outputs, decoder_hidden, decoder_inputs
                )
                decoder_outputs.append(output)

                if targets is not None:
                    # Teacher forcing: Feed the target as the next input
                    decoder_inputs = targets[:, i].unsqueeze(1)  # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    _, topi = output.topk(1)
                    decoder_inputs = topi.squeeze(
                        -1
                    ).detach()  # detach from history as input

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            return decoder_outputs

    def __init__(self, args):
        super(Seq2SeqAttn, self).__init__()
        self.epoch = 0
        self.args = args
        self.device = args["device"]

        self.encoder = Seq2SeqAttn.Encoder(args)
        self.decoder = Seq2SeqAttn.Decoder(args)

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
    scheduler: Optional[Any] = None,
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

    if scheduler is not None:
        scheduler.step()

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
