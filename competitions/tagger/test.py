import torch
import torch.nn as nn

from Morpho import MorphoDataset
from transformer import Transformer

morpho = MorphoDataset("czech_pdt")
args = {
    "batch_size": 2,
    "epochs": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "Seq2Seq",
    "model_dim": 512,
    "max_seq_len": morpho.max_length,
    "encoder_stack_size": 1,
    "decoder_stack_size": 1,
    "word_vocab_size": morpho.train.unique_forms,
    "num_classes": morpho.train.unique_tags,
    "label_smoothing": 0.1,
}

model = Transformer(args)

train_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)
optim = torch.optim.AdamW(model.parameters())
loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
train_dloader = morpho.train.to_dataloader(args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)

model.train()
for batch in train_dloader:
    words = batch["words"].to(model.device)
    tags = batch["tags"].to(model.device)
    break
#     chars = batch["chars"]
#     tags = batch["tags"].to(model.device)
#     words_num = batch["words_num"].to(model.device)

#     max_words_num = torch.max(words_num)
#     mask = torch.arange(max_words_num, device=model.device).expand(
#         len(words_num), max_words_num
#     ) < words_num.unsqueeze(1)

#     # Run inference
#     y_hat = model(words, batch["words_num"].to(model.device), chars, mask, tags)
#     loss = loss_fn(y_hat[mask], tags[mask])

#     # Update params
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     break
