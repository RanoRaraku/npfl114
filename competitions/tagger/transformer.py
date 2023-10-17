import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, device: str = "cpu"):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ScaledDotAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dk: int,
        dv: int,
        apply_Wo: bool = True,
        device: str = "cpu",
    ):
        super(ScaledDotAttention, self).__init__()
        self.dk = torch.tensor(dk, dtype=torch.float32, device=device)
        if apply_Wo:
            self.Wo = nn.Linear(dv, model_dim, device=device)
        else:
            self.Wo = nn.Identity()
        self.device = device

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        apply_mask: bool = False,
    ) -> torch.Tensor:
        B = keys.shape[0]
        scores = torch.bmm(query, keys.permute(0, 2, 1)) / torch.sqrt(self.dk)
        if apply_mask:
            T = query.shape[1]
            mask = torch.triu(
                torch.ones(B, T, T, dtype=torch.bool, device=self.device), 1
            )
            scores[mask] = float("-inf")
        weights = nn.functional.softmax(scores, -1)
        context = torch.bmm(weights, values)
        context = self.Wo(context)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(
        self, model_dim: int, dk: int, dv: int, heads: int, device: str = "cpu"
    ):
        super(MultiHeadAttention, self).__init__()
        assert dk % heads == 0
        assert dv % heads == 0

        self.att_list = nn.ModuleList(
            [
                ScaledDotAttention(
                    model_dim,
                    int(dk / heads),
                    int(dv / heads),
                    apply_Wo=False,
                    device=device,
                )
                for _ in range(heads)
            ]
        )
        self.Wo = nn.Linear(dv, model_dim, device=device)

    def forward(
        self,
        query: list[torch.Tensor],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        apply_mask: bool = False,
    ) -> torch.Tensor:
        context = [
            att(q, k, v, apply_mask)
            for att, q, k, v in zip(self.att_list, query, keys, values)
        ]
        context = self.Wo(torch.cat(context, dim=-1))
        return context


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len: int, pe_dim: int, device: str = "cpu"):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class InputLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        max_seq_len: int,
        dropout_p: float = 0.2,
        device: str = "cpu",
    ):
        super(InputLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_dim, model_dim, device=device)
        self.position_encoding = PositionEncoding(max_seq_len, model_dim, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        return x


class AttentionProjection(nn.Module):
    def __init__(self, model_dim: int, dk: int, dv: int, device: str = "cpu"):
        super(AttentionProjection, self).__init__()
        self.Wq = nn.Linear(model_dim, dk, device=device)
        self.Wk = nn.Linear(model_dim, dk, device=device)
        self.Wv = nn.Linear(model_dim, dv, device=device)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.Wq(x)
        K = self.Wk(y if y is not None else x)
        V = self.Wv(z if z is not None else (y if y is not None else x))
        return q, K, V


class MultiAttentionProjection(nn.Module):
    def __init__(
        self, model_dim: int, dk: int, dv: int, heads: int, device: str = "cpu"
    ):
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

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        query, keys, values = [], [], []
        for Wq, Wk, Wv in zip(self.Wq, self.Wk, self.Wv):
            query.append(Wq(x))
            keys.append(Wk(y if y is not None else x))
            values.append(Wv(z if z is not None else (y if y is not None else x)))

        return query, keys, values


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        keys_dim: int,
        values_dim: int,
        attention_heads: int = 1,
        device: str = "cpu",
    ) -> None:
        super(Encoder, self).__init__()

        self.selfatt_projection = MultiAttentionProjection(
            model_dim, keys_dim, values_dim, attention_heads, device
        )
        self.selfatt = MultiHeadAttention(
            model_dim, keys_dim, values_dim, attention_heads, device
        )
        self.selfatt_norm = nn.LayerNorm(model_dim, device=device)
        self.ffn = FFN(model_dim, device=device)
        self.ffn_norm = nn.LayerNorm(model_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, K, V = self.selfatt_projection(x)
        c = self.selfatt(q, K, V, False)
        x = self.selfatt_norm(x + c)
        y = self.ffn(x)
        x = self.ffn_norm(x + y)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        keys_dim: int,
        values_dim: int,
        attention_heads: int = 1,
        device: str = "cpu",
    ):
        super(Decoder, self).__init__()
        self.dk = keys_dim
        self.dv = values_dim

        self.selfatt_projection = MultiAttentionProjection(
            model_dim, self.dk, self.dv, attention_heads, device
        )
        self.selfatt = MultiHeadAttention(
            model_dim, self.dk, self.dv, attention_heads, device
        )
        self.selfatt_norm = nn.LayerNorm(model_dim, device=device)

        self.edatt_projection = MultiAttentionProjection(
            model_dim, self.dk, self.dv, attention_heads, device
        )
        self.edatt = MultiHeadAttention(
            model_dim, self.dk, self.dv, attention_heads, device
        )
        self.edatt_norm = nn.LayerNorm(model_dim, device=device)

        self.ffn = FFN(model_dim, device=device)
        self.ffn_norm = nn.LayerNorm(model_dim, device=device)

    def forward(
        self, inputs: torch.Tensor, encodings: torch.Tensor, apply_mask: bool = False
    ) -> torch.Tensor:
        q, K, V = self.selfatt_projection(inputs)
        x = self.selfatt(q, K, V, apply_mask)
        x = self.selfatt_norm(x + inputs)

        q, K, V = self.edatt_projection(x, encodings)
        y = self.edatt(q, K, V, False)
        y = self.edatt_norm(y + x)

        z = self.ffn(y)
        z = self.ffn_norm(z + y)

        return z


class Transformer(nn.Module):
    def __init__(self, args: dict):
        super(Transformer, self).__init__()

        self.args = args
        self.device = args["device"]
        self.epoch = 0

        # Encoder
        self.encoder_input = InputLayer(
            args["input_vocab_size"],
            args["model_dim"],
            args["max_seq_len"],
            args["input_dropout"],
            self.device,
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
        self.decoder_input = InputLayer(
            args["num_classes"],
            args["model_dim"],
            args["max_seq_len"],
            args["input_dropout"],
            self.device,
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

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        max_seq_len: int = 50,
    ) -> torch.Tensor:
        # Encoder
        e = self.encoder_input(inputs)
        e = self.encoder_stack(e)

        # Decoder
        if outputs is not None:
            # TRAIN: teacher forcing with self_att causal mask
            d = self.decoder_input(outputs)
            for decoder in self.decoder_stack:
                d = decoder(d, e, True)
            out = self.out(d)
        else:
            # INFERENCE: auto-regressive decoding
            batch_size = inputs.size(0)
            decoder_inputs = torch.zeros(
                batch_size, 1, dtype=torch.long, device=self.device
            )

            for _ in range(max_seq_len):
                d = self.decoder_embedding(decoder_inputs)
                d = self.position_encoding(d)
                for decoder in self.decoder_stack:
                    d = decoder(d, e, True)
                out = self.out(d)
                next_token = self.sample_token(out[:, -1])
                decoder_inputs = torch.cat((decoder_inputs, next_token), dim=1)
        return out

    def sample_token(self, token_pdf: torch.Tensor):
        _, topi = token_pdf.topk(1)
        return topi.detach()


def eval_accuracy(model, dloader):
    """
    Returns accuracy.

    :dloader: torch.utils.data.DataLoader object
    """
    model.eval()
    total_loss, total_samples, corr = 0, 0, 0
    with torch.no_grad():
        for batch in dloader:
            words = batch["words"].to(model.device)
            tags = batch["tags"].to(model.device)
            words_num = batch["words_num"].to(model.device)

            max_words_num = torch.max(words_num)
            mask = torch.arange(max_words_num, device=model.device).expand(
                len(words_num), max_words_num
            ) < words_num.unsqueeze(1)

            # Run inference
            y_hat = model(words, words_num)
            corr += torch.sum(torch.argmax(y_hat[mask], dim=-1) == tags[mask])
            total_samples += torch.sum(words_num)

    return corr / total_samples, total_loss / len(dloader)


def train_epoch(
    model,
    train_dataloader,
    dev_dataloader,
    loss_fn,
    optim,
    lr_scheduler: Optional[Callable] = None,
    logger: Optional[Callable] = None,
):
    """
    :train_dataloader: a torch.utils.data.DataLoader object
    :dev_dataloader: a torch.utils.data.DataLoader object
    :loss_fn: a callable loss function
    :optim: a torch.optim object
    :lr_scheduler: a learning rate scheduler
    :logger: a callable logger (i.e wandb)
    """

    # TRAIN on train
    start_time = time.time()
    model.train()
    for batch in train_dataloader:
        words = batch["words"].to(model.device)
        tags = batch["tags"].to(model.device)
        words_num = batch["words_num"].to(model.device) - 1

        max_words_num = torch.max(words_num)
        mask = torch.arange(max_words_num, device=model.device).expand(
            len(words_num), max_words_num
        ) < words_num.unsqueeze(1)

        # Run inference
        input_targets = tags[:, :-1]
        output_targets = tags[:, 1:]
        y_hat = model(words, input_targets)
        loss = loss_fn(y_hat[mask], output_targets[mask])

        # Update params
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss.item())
        exit()

        if logger is not None:
            logger.log({"train_loss": loss.item()})

    # EVAL on dev
    dev_samples, dev_corr, dev_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dev_dataloader:
            words = batch["words"].to(model.device)
            tags = batch["tags"].to(model.device)
            words_num = batch["words_num"].to(model.device) - 1

            max_words_num = torch.max(words_num)
            mask = torch.arange(max_words_num, device=model.device).expand(
                len(words_num), max_words_num
            ) < words_num.unsqueeze(1)

            # Run inference
            input_targets = tags[:, :-1]
            output_targets = tags[:, 1:]
            y_hat = model(words, input_targets)
            loss = loss_fn(y_hat[mask], output_targets[mask])

            dev_loss += loss.detach().item()
            dev_corr += torch.sum(
                torch.argmax(y_hat[mask], dim=-1) == output_targets[mask]
            )
            dev_samples += torch.sum(words_num)
    dev_acc = dev_corr / dev_samples
    dev_loss /= len(dev_dataloader)
    end_time = time.time()

    # Log
    model.epoch += 1
    if lr_scheduler is not None:
        lr_scheduler.step()
    if logger is not None:
        logger.log(
            {
                "epoch_time": end_time - start_time,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }
        )


def train_transformer(
    model,
    train_dataloader,
    loss_fn,
    optim,
    args,
):
    """
    :train_dataloader: a torch.utils.data.DataLoader object
    :dev_dataloader: a torch.utils.data.DataLoader object
    :loss_fn: a callable loss function
    :optim: a torch.optim object
    :lr_scheduler: a learning rate scheduler
    :logger: a callable logger (i.e wandb)
    """
    model = torch.nn.Transformer(
        512, 8, 2, 2, 2048, 0.1, "relu", batch_first=True, device="cpu"
    )

    encoder_embedding = nn.Embedding(args["input_vocab_size"], args["model_dim"])
    position_encoding = PositionEncoding(args["max_seq_len"], args["model_dim"])
    decoder_embedding = nn.Embedding(args["num_classes"], args["model_dim"])
    out = nn.Linear(args["model_dim"], args["num_classes"])

    model.train()
    for batch in train_dataloader:
        words = batch["words"]
        tags = batch["tags"]
        words_num = batch["words_num"]

        max_words_num = torch.max(words_num)
        mask = torch.arange(max_words_num).expand(
            len(words_num), max_words_num
        ) < words_num.unsqueeze(1)

        e = encoder_embedding(words)
        e = position_encoding(e)
        d = decoder_embedding(tags)
        d = position_encoding(d)

        # Run inference
        y_hat = model(e, d)
        y_hat = out(y_hat)
        loss = loss_fn(y_hat[mask], tags[mask])

        # Update params
        optim.zero_grad()
        loss.backward()
        optim.step()
