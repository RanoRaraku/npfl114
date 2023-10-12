import time
from typing import Any, Optional

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, device="cpu") -> None:
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ScaledDotAttention(nn.Module):
    def __init__(self, model_dim, dk, dv, device="cpu"):
        super(ScaledDotAttention, self).__init__()
        self.dk = torch.tensor(dk, dtype=torch.float32, device=device)
        self.Wo = nn.Linear(dv, model_dim, device=device)
        self.device = device

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        apply_mask: bool = False,
        apply_Wo: bool = True,
    ):
        scores = torch.bmm(query, keys.permute(0, 2, 1)) / torch.sqrt(self.dk)
        if apply_mask:
            T = query.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=self.device), 1)
            scores += -1e12 * mask
        weights = nn.functional.softmax(scores, -1)
        context = torch.bmm(weights, values)
        if apply_Wo:
            context = self.Wo(context)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, dk, dv, heads, device="cpu"):
        super(MultiHeadAttention, self).__init__()
        assert dk % heads == 0
        assert dv % heads == 0

        self.att_list = nn.ModuleList(
            [
                ScaledDotAttention(model_dim, int(dk / heads), int(dv / heads), device)
                for _ in range(heads)
            ]
        )
        self.Wo = nn.Linear(dv, model_dim, device=device)

    def forward(
        self,
        query: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        mask: bool = False,
    ):
        context = [
            att(q, k, v, mask, False)
            for att, q, k, v in zip(self.att_list, query, keys, values)
        ]
        context = self.Wo(torch.cat(context, dim=-1))
        return context


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len, pe_dim, device="cpu"):
        """
        Pre-computed for speed, figure out actual x_length during inference.

        max_seq_len: max number of words in a sequence
        dim: position encoding dim which equals last dimension of input
        """
        super(PositionEncoding, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim = pe_dim

        assert pe_dim % 2 == 0

        d = int(pe_dim / 2)
        self.pe = torch.empty(max_seq_len, pe_dim, dtype=torch.float32, device=device)
        for k in range(max_seq_len):
            g = k / (10000 ** (2 * torch.arange(d) / pe_dim))
            self.pe[k, :d] = torch.sin(g)
            self.pe[k, d:] = torch.cos(g)

    def forward(self, x):
        """
        Expect `x` to be tensor of shape [T,C] or [B,T,C].
        """
        if x.dim() == 2:
            k_dim = 0
            i_dim = 1
        elif x.dim() == 3:
            k_dim = 1
            i_dim = 2

        assert x.shape[k_dim] <= self.max_seq_len
        assert x.shape[i_dim] == self.dim
        xk_dim = x.shape[k_dim]

        x = x + self.pe[:xk_dim, :]
        return x


class AttentionProjection(nn.Module):
    def __init__(self, model_dim, dk, dv, device="cpu"):
        super(AttentionProjection, self).__init__()
        self.Wq = nn.Linear(model_dim, dk, device=device)
        self.Wk = nn.Linear(model_dim, dk, device=device)
        self.Wv = nn.Linear(model_dim, dv, device=device)

    def forward(self, x, y=None, z=None):
        q = self.Wq(x)
        K = self.Wk(y if y is not None else x)
        V = self.Wv(z if z is not None else (y if y is not None else x))
        return q, K, V


class MultiAttentionProjection(nn.Module):
    def __init__(self, model_dim, dk, dv, heads, device="cpu"):
        super(MultiAttentionProjection, self).__init__()
        assert dk % heads == 0
        assert dv % heads == 0

        self.Wq = nn.ModuleList(
            [nn.Linear(model_dim, int(dk / heads), device=device) for _ in range(heads)]
        )
        self.Wk = nn.ModuleList(
            [nn.Linear(model_dim, int(dk / heads), device=device) for _ in range(heads)]
        )
        self.Wv = nn.ModuleList(
            [nn.Linear(model_dim, int(dv / heads), device=device) for _ in range(heads)]
        )

    def forward(self, x, y=None, z=None):
        query, keys, values = [], [], []
        for Wq, Wk, Wv in zip(self.Wq, self.Wk, self.Wv):
            query.append(Wq(x))
            keys.append(Wk(y if y is not None else x))
            values.append(Wv(z if z is not None else (y if y is not None else x)))

        return query, keys, values


class Encoder(nn.Module):
    def __init__(self, model_dim, keys_dim, values_dim, heads=1, device="cpu") -> None:
        super(Encoder, self).__init__()

        self.selfatt_projection = MultiAttentionProjection(
            model_dim, keys_dim, values_dim, heads, device
        )
        self.selfatt = MultiHeadAttention(
            model_dim, keys_dim, values_dim, heads, device
        )
        self.selfatt_norm = nn.LayerNorm(model_dim, device=device)
        self.ffn = FFN(model_dim, device=device)
        self.ffn_norm = nn.LayerNorm(model_dim, device=device)

    def forward(self, x):
        q, K, V = self.selfatt_projection(x)
        c = self.selfatt(q, K, V)
        x = self.selfatt_norm(x + c)
        y = self.ffn(x)
        x = self.ffn_norm(x + y)

        return x


class Decoder(nn.Module):
    def __init__(self, model_dim, keys_dim, values_dim, heads=1, device="cpu") -> None:
        super(Decoder, self).__init__()
        self.dk = keys_dim
        self.dv = values_dim

        self.selfatt_projection = MultiAttentionProjection(
            model_dim, self.dk, self.dv, heads, device
        )
        self.selfatt = MultiHeadAttention(model_dim, self.dk, self.dv, heads, device)
        self.selfatt_norm = nn.LayerNorm(model_dim, device=device)

        self.edatt_projection = MultiAttentionProjection(
            model_dim, self.dk, self.dv, heads, device
        )
        self.edatt = MultiHeadAttention(model_dim, self.dk, self.dv, heads, device)
        self.edatt_norm = nn.LayerNorm(model_dim, device=device)

        self.ffn = FFN(model_dim, device=device)
        self.ffn_norm = nn.LayerNorm(model_dim, device=device)

    def forward(self, inputs, encodings):
        q, K, V = self.selfatt_projection(inputs)
        x = self.selfatt(q, K, V, True)
        x = self.selfatt_norm(x + inputs)

        q, K, V = self.edatt_projection(x, encodings)
        y = self.edatt(q, K, V)
        y = self.edatt_norm(y + x)

        z = self.ffn(y)
        z = self.ffn_norm(z + y)

        return z


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super(Transformer, self).__init__()

        self.args = args
        self.device = args["device"]
        self.epoch = 0

        # Encoder
        self.inputs_embedding = nn.Embedding(
            args["word_vocab_size"], args["model_dim"], device=self.device
        )
        self.position_encoding = PositionEncoding(
            args["max_seq_len"], args["model_dim"], self.device
        )
        self.encoder_stack = nn.Sequential(
            *[
                Encoder(
                    args["model_dim"],
                    args["keys_dim"],
                    args["values_dim"],
                    args["heads"],
                    self.device,
                )
                for _ in range(args["encoder_stack_size"])
            ]
        )

        # Decoder
        self.outputs_embedding = nn.Embedding(
            args["num_classes"], args["model_dim"], device=self.device
        )
        self.decoder_stack = nn.ModuleList(
            [
                Decoder(
                    args["model_dim"],
                    args["keys_dim"],
                    args["values_dim"],
                    args["heads"],
                    self.device,
                )
                for _ in range(args["decoder_stack_size"])
            ]
        )

        self.out = nn.Linear(args["model_dim"], args["num_classes"], device=self.device)

    def forward(self, inputs, outputs):
        e = self.inputs_embedding(inputs)
        e = self.position_encoding(e)
        e = self.encoder_stack(e)

        d = self.outputs_embedding(outputs)
        d = self.position_encoding(d)
        for decoder in self.decoder_stack:
            d = decoder(d, e)

        y = self.out(d)

        return y


def eval_accuracy(model, dloader, loss_fn: Optional[Any] = None):
    """
    Returns accuracy and optionally per_sample loss.
    """
    model.eval()
    total_loss, total_samples, corr = 0, 0, 0
    for batch in dloader:
        words = batch["words"].to(model.device)
        tags = batch["tags"].to(model.device)
        words_num = batch["words_num"].to(model.device)

        max_words_num = torch.max(words_num)
        mask = torch.arange(max_words_num, device=model.device).expand(
            len(words_num), max_words_num
        ) < words_num.unsqueeze(1)

        # Run inference
        y_hat = model(words, tags)
        corr += torch.sum(torch.argmax(y_hat[mask], dim=-1) == tags[mask])
        total_samples += torch.sum(words_num)

        if loss_fn:
            loss = loss_fn(y_hat[mask], tags[mask])
            total_loss += loss.item()

    return corr / total_samples, total_loss / len(dloader)


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
        words = batch["words"].to(model.device)
        tags = batch["tags"].to(model.device)
        words_num = batch["words_num"].to(model.device)

        max_words_num = torch.max(words_num)
        mask = torch.arange(max_words_num, device=model.device).expand(
            len(words_num), max_words_num
        ) < words_num.unsqueeze(1)

        # Run inference
        y_hat = model(words, tags)
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
