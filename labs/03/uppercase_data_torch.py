import os
import sys
from typing import Dict, List, TextIO, Union
import urllib.request
import zipfile
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Loads the Uppercase data.
# - The data consists of three Datasets
#   - `train`
#   - `dev`
#   - `test` [all in lowercase]
# - When loading, you need to specify `window` and `alphabet_size`. If
#   `alphabet_size` is nonzero, it specifies the maximum number of alphabet
#   characters, in which case that many most frequent characters will be used,
#   and all other will be remapped to "<unk>".
# - Features are generated using a sliding window of given size,
#   i.e., for a character, we include left `window` characters, the character
#   itself and right `window` characters, `2 * window + 1` in total.
# - Each dataset (train/dev/test) has the following members:
#   - `size`: the length of the text
#   - `data`: a dictionary with keys
#       - "windows": input examples with shape `[size, 2 * window_size + 1]`,
#            corresponding to indices of input lowercased characters
#       - "labels": input labels with shape `[size]`, each a 0/1 value whether
#            the corresponding input in `windows` is lowercased/uppercased
#   - `text`: the original text (of course lowercased in case of the test set)
#   - `alphabet`: an alphabet used by `windows`
#   - `dataset`: a PyTorch `tf.utils..Dataset` producing as examples dictionaries
#       with keys "windows" and "labels"


class DataFromDict(Dataset):
    def __init__(self,input_dict ):
        self.input_dict = input_dict
        self.input_keys = list(input_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self,idx):
        item = self.input_dict[self.img_keys[idx]]['item_key']
        label = self.input_dict[self.img_keys[idx]]['label_key']
        return item, label


class UppercaseData:
    LABELS: int = 2

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/uppercase_data.zip"

    class CharDataset(Dataset):
        def __init__(self, data: str, window: int, alphabet: Union[int, List[str]], seed: int = 42) -> None:
            self._window = window
            self._text = data
            self._size = len(self._text)

            # Create alphabet_map
            if isinstance(alphabet, list):
                alphabet_map = {}
                for index, letter in enumerate(alphabet):
                    alphabet_map[letter] = index
            else:
                # Find most frequent characters
                alphabet_map = {"<pad>": 0, "<unk>": 1}
                freqs = {}
                for char in self._text.lower():
                    freqs[char] = freqs.get(char, 0) + 1

                most_frequent = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
                for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                    alphabet_map[char] = i
                    if alphabet and len(alphabet_map) >= alphabet:
                        break

            # Compute alphabet
            self.alphabet_map = alphabet_map
            self._alphabet = [None] * len(alphabet_map)
            for key, value in alphabet_map.items():
                self._alphabet[value] = key

            # Remap lowercased input characters using the alphabet_map
            lcletters = np.zeros(self._size + 2 * window, np.int16)
            for i in range(self._size):
                char = self._text[i].lower()
                if char not in alphabet_map:
                    char = "<unk>"
                lcletters[i + window] = alphabet_map[char]

            # Generate input batches
            samples = torch.zeros(self._size, 2 * window + 1, dtype=torch.int16)
            labels = torch.zeros([self._size, 1], dtype=torch.int16)
            for i in range(self._size):
                samples[i] = torch.from_numpy(lcletters[i:i + 2 * window + 1])
                labels[i] = torch.as_tensor(1 * self._text[i].isupper(), dtype=torch.int16)
            self._samples = samples
            self._labels = labels

        def __len__(self):
            return len(self._labels)

        def __getitem__(self, idx):
            return {
                "idx":idx,
                "sample":torch.nn.functional.one_hot(self._samples[idx].to(torch.int64),len(self.alphabet)).to(torch.float32),
                "label":self._labels[idx].to(torch.float32)
            }

        @property
        def alphabet(self) -> List[str]:
            return self._alphabet

        @property
        def text(self) -> str:
            return self._text

        @property
        def size(self) -> int:
            return self._size


    def __init__(self, window: int, alphabet_size: int = 0):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    data = dataset_file.read().decode("utf-8")
                setattr(self, dataset, self.CharDataset(
                    data,
                    window,
                    alphabet=alphabet_size if dataset == "train" else self.train.alphabet,
                ))

    train: CharDataset
    dev: CharDataset
    test: CharDataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: CharDataset, predictions: str) -> float:
        gold = gold_dataset.text

        if len(predictions) < len(gold):
            raise RuntimeError("The predictions are shorter than gold data: {} vs {}.".format(
                len(predictions), len(gold)))

        correct = 0
        for i in range(len(gold)):
            # Note that just the lower() condition is not enough, for example
            # u03c2 and u03c3 have both u03c2 as an uppercase character.
            if predictions[i].lower() != gold[i].lower() and predictions[i].upper() != gold[i].upper():
                raise RuntimeError("The predictions and gold data differ on position {}: {} vs {}.".format(
                    i, repr(predictions[i:i + 20].lower()), repr(gold[i:i + 20].lower())))

            correct += gold[i] == predictions[i]
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: CharDataset, predictions_file: TextIO) -> float:
        predictions = predictions_file.read()
        return UppercaseData.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = UppercaseData.evaluate_file(getattr(UppercaseData(0), args.dataset), predictions_file)
        print("Uppercase accuracy: {:.2f}%".format(accuracy))
