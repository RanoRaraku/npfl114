import os
import sys
import zipfile
from typing import Any, BinaryIO, Optional, Tuple

import torch
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Loads a morphological dataset 'MorphoDataset'
# - The morphological dataset consists of two subdatasets 'CustomDataset'
#   - `train`
#   - `dev`
# - Each 'CustomDataset' is of PyTorch format and composed of
#   - `size`: number of sentences in the dataset
#   - `forms`, `lemmas`, `tags`: objects containing the following fields:
#     - `strings`: a Python list containing input sentences, each being
#         a list of strings (forms/lemmas/tags)
#     - `word_mapping`: a `tf.keras.layers.StringLookup` object capable of
#         mapping words to indices. It is constructed on the train set and
#         shared by the dev and test sets.
#     - `char_mapping`: a `tf.keras.layers.StringLookup` object capable of
#         mapping characters to indices. It is constructed on the train set
#         and shared by the dev and test sets.
#   - max_length - maximum length of a sequence (sentence)
#   - methods to convert it to a DataLoader and a collate() function



class Factor:
    def __init__(self) -> None:
        self.strings = []
        self.word_mapping = {
            "<BOS>": 0,
            "<EOS>": 1,
        }
        self.char_mapping = {
            "<BOS>": 0,
            "<EOS>": 1,
        }

    def finalize(self, dev: Optional[Any] = None):
        # some super long senteces, thats suspicious
        self.strings = [s for s in self.strings if len(s) < 50]

        self.chars = [
            [["<BOS>"]] + [[char for char in word] for word in sentence] + [["<EOS>"]]
            for sentence in self.strings
        ]
        char_vocab = [
            char for sentence in self.chars for word in sentence for char in word
        ]

        self.strings = [["<BOS>"] + sentence + ["<EOS>"] for sentence in self.strings]
        word_vocab = [word for sentence in self.strings for word in sentence]

        if dev:
            word_vocab += [word for sentence in dev.strings for word in sentence]
            char_vocab += [
                char for sentence in dev.chars for word in sentence for char in word
            ]

        self.char_mapping.update(
            {
                k: v
                for v, k in enumerate(sorted(set(char_vocab)))
                if k not in ["<BOS>", "<EOS>"]
            }
        )
        self.word_mapping.update(
            {
                k: v
                for v, k in enumerate(sorted(set(word_vocab)))
                if k not in ["<BOS>", "<EOS>"]
            }
        )


class CustomDataset(Dataset):
    """ """

    def __init__(
        self,
        data_file: BinaryIO,
        dev: Optional[Any] = None,
        max_sentences: Optional[int] = None,
    ):
        # Create factors
        self._factors = Factor(), Factor(), Factor()

        # Load the data
        in_sentence = False
        for line in data_file:
            line = line.decode("utf-8").rstrip("\r\n")
            if line:
                if not in_sentence:
                    for factor in self._factors:
                        factor.strings.append([])

                columns = line.split("\t")
                assert len(columns) == len(self._factors)
                for column, factor in zip(columns, self._factors):
                    factor.strings[-1].append(column)
                in_sentence = True
            else:
                in_sentence = False
                if max_sentences is not None and self._size >= max_sentences:
                    break

        # Finalize the mappings
        for i, factor in enumerate(self._factors):
            factor.finalize(dev._factors[i] if dev else None)
        self._size = len(self.forms.strings)

        self.unique_words = len(self.forms.word_mapping)
        self.words_bos = self.forms.word_mapping["<BOS>"]
        self.words_eos = self.forms.word_mapping["<EOS>"]

        self.unique_chars = len(self.forms.char_mapping)
        self.chars_bos = self.forms.char_mapping["<BOS>"]
        self.chars_eos = self.forms.char_mapping["<EOS>"]

        self.unique_tags = len(self.tags.word_mapping)
        self.tags_bos = self.tags.word_mapping["<BOS>"]
        self.tags_eos = self.tags.word_mapping["<EOS>"]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        words = tensor(
            [self.forms.word_mapping[word] for word in self.forms.strings[index]]
        ).to(torch.int32)
        chars = [
            tensor([self.forms.char_mapping[c] for c in w]).to(torch.int32)
            for w in self.forms.chars[index]
        ]
        tags = tensor([self.tags.word_mapping[tag] for tag in self.tags.strings[index]])

        return words, chars, tags

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def forms(self) -> Factor:
        return self._factors[0]

    @property
    def lemmas(self) -> Factor:
        return self._factors[1]

    @property
    def tags(self) -> Factor:
        return self._factors[2]

    def collate(self, samples):
        words, chars, tags = zip(*samples)
        words_num = tensor(list(map(len, words))).to(torch.int64)
        max_len = torch.max(words_num).item()
        chars = [
            ct + [tensor([self.chars_eos]).to(torch.int32) for _ in range(max_len - wn)]
            for ct, wn in zip(chars, words_num)
        ]
        return {
            "words": pad_sequence(
                words, batch_first=True, padding_value=self.words_eos
            ),
            "words_num": words_num,
            "chars": chars,
            "tags": pad_sequence(tags, batch_first=True, padding_value=self.tags_eos),
        }

    def to_dataloader(
        self, batch_size: int = 128, shuffle: bool = True, **kwargs
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            collate_fn=self.collate,
            shuffle=shuffle,
            **kwargs,
        )


class MorphoDataset:
    BOS: int = 0
    EOS: int = 1
    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/"

    train: CustomDataset
    dev: CustomDataset

    def __init__(self, dataset, max_sentences=None):
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Missing dataset {}...".format(dataset), file=sys.stderr)
            exit()

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["dev", "train"]:
                with zip_file.open(
                    "{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r"
                ) as dataset_file:
                    setattr(
                        self,
                        dataset,
                        CustomDataset(
                            dataset_file,
                            dev=self.dev if dataset == "train" else None,
                            max_sentences=max_sentences,
                        ),
                    )

        # hack to share mappings
        self.dev.forms.word_mapping = self.train.forms.word_mapping
        self.dev.forms.char_mapping = self.train.forms.char_mapping
        self.dev.tags.word_mapping = self.train.tags.word_mapping
        self.dev.tags.char_mapping = self.train.tags.char_mapping
        self.dev.unique_words = self.train.unique_words
        self.dev.unique_tags = self.train.unique_tags

        # set maximum sequence length
        self.max_length = max(
            [len(s) for s in self.train.forms.strings]
            + [len(s) for s in self.dev.forms.strings]
        )
