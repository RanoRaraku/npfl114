import os
import sys
from typing import BinaryIO, Optional, Dict, Tuple, Any
import zipfile

import torch
from torch import tensor, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence

# Loads a morphological dataset in a vertical format.
# - The data consists of three datasets
#   - `train`
#   - `dev`
#   - `test`
# - Each dataset is composed of
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
#   - `dataset`: a `tf.data.Dataset` containing a dictionary with "forms", "lemmas", "tags".


class Factor:
    word_mapping: Dict
    char_mapping: Dict

    def __init__(self) -> None:
        self.strings = []

    def finalize(self, dev: Optional[Any] = None, add_bow_eow: bool = False) -> None:

        word_vocab = [string for sentence in self.strings for string in sentence]

        additional_characters = []
        if add_bow_eow:
            additional_characters.extend(["[BOW]", "[EOW]"])
        char_vocab = additional_characters + [char for sentence in self.strings for string in sentence for char in string]

        if dev:
            word_vocab += [string for sentence in dev.strings for string in sentence]
            char_vocab += [char for sentence in dev.strings for string in sentence for char in string]

        self.char_mapping = {k:v for v,k in enumerate(sorted(set(char_vocab)))}
        self.word_mapping = {k:v for v,k in enumerate(sorted(set(word_vocab)))}


class CustomDataset(Dataset):

    def __init__(
        self,
        data_file: BinaryIO,
        dev: Optional[Any] = None,
        max_sentences: Optional[int] = None,
        add_bow_eow: bool = False
    ):
        # Create factors
        self._factors = Factor(), Factor(), Factor()

        # Load the data
        self._size = 0
        in_sentence = False
        for line in data_file:
            line = line.decode("utf-8").rstrip("\r\n")
            if line:
                if not in_sentence:
                    for factor in self._factors:
                        factor.strings.append([])
                    self._size += 1

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
            factor.finalize(dev._factors[i] if dev else None, add_bow_eow)

        self.unique_forms = len(self.forms.word_mapping)
        self.unique_tags = len(self.tags.word_mapping)


    def __getitem__(self, index:int) -> Tuple[Tensor, Tensor]:
        words = tensor([self.forms.word_mapping[word] for word in self.forms.strings[index]]).to(torch.long)
        chars = tensor([self.forms.char_mapping[char] for word in self.forms.strings[index] for char in word]).to(torch.long)
        chars_lens = tensor(list(map(len, chars)))

        tags = tensor([self.tags.word_mapping[tag] for tag in self.tags.strings[index]])
        tags = one_hot(tags, self.unique_tags).to(torch.float)

        return words, chars, chars_lens, tags


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

    @staticmethod
    def collate(samples):
        words, chars, chars_lens, tags = zip(*samples)
        seq_lens = tensor(list(map(len, words)))
        seq_lens, perm_idx = seq_lens.sort(0, descending=True)
        return {
            "words": pad_sequence(words[perm_idx], batch_first=True),
            "chars": pad_sequence(chars[perm_idx], batch_first=True),
            "tags": pad_sequence(tags[perm_idx], batch_first=True),
            "sequence_lens": seq_lens,
            "chars_lens": chars_lens[perm_idx],
        }

    def to_dloader(self, batch_size:int=128, shuffle:bool=True, **kwargs) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            collate_fn=self.collate,
            shuffle=shuffle,
            **kwargs,
        )
    

class MorphoDataset:
    BOW: int = 1
    EOW: int = 2
    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/"

    def __init__(self, dataset, max_sentences=None, add_bow_eow=False):
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Missing dataset {}...".format(dataset), file=sys.stderr)
            exit

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["dev", "train"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.CustomDataset(
                        dataset_file, dev=self.dev if dataset == "train" else None,
                        max_sentences=max_sentences, add_bow_eow=add_bow_eow)
                    )
        # hack to share mappings
        self.dev.forms.word_mapping = self.train.forms.word_mapping
        self.dev.forms.char_mapping = self.train.forms.char_mapping
        self.dev.tags.word_mapping = self.train.tags.word_mapping
        self.dev.tags.char_mapping = self.train.tags.char_mapping
        self.dev.unique_forms = self.train.unique_forms
        self.dev.unique_tags = self.train.unique_tags

    train: CustomDataset
    dev: CustomDataset

