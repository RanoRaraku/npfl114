import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence


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
        y = [[self.char_embedd(word.to(self.device)) for word in sent] for sent in chars]
        # pack words in sentence into one packed sequence
        y = [pack_sequence(embeddings, enforce_sorted=False) for embeddings in y]
        # output hidden state h_n for last character in a word
        y = [self.char_lstm(packed)[1][0].squeeze(0) for packed in y]
        # pad to match dimensionality of word_lstm
        y = pad_sequence(y, batch_first=True)

        z = self.linear(x + y)
        return z

class Seq2Seq(nn.Module):

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
                hidden_size=2*args["hidden_size"],
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
            y = [[self.char_embedd(word.to(self.device)) for word in sent] for sent in chars]
            # pack words in sentence into one packed sequence
            y = [pack_sequence(embeddings, enforce_sorted=False) for embeddings in y]
            # output hidden state h_n for last character in a word
            y = [self.char_lstm(packed)[1][0].squeeze(0) for packed in y]
            # pad to match dimensionality of word_lstm
            y = pad_sequence(y, batch_first=True)

            z = x + y
            return z

    class Decoder(nn.Module):
        def __init__(self, args):
            super(Seq2Seq.Decoder, self).__init__()
            self.epoch = 0
            self.args = args
            self.device = args["device"]

        def forward(self):
            ...

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.epoch = 0
        self.args = args
        self.device = args["device"]

        self.encoder = Seq2Seq.Encoder(args)


    def forward(self, words, words_num, chars):
        
        print(words.shape)
        encoded = self.encoder(words, words_num, chars)
        print(encoded.shape)
        
