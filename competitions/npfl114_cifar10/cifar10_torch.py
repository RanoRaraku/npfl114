import os
import sys
from typing import Dict, List, Sequence, TextIO
import urllib.request
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torchvision import transforms

class CIFAR10:
    H: int = 32
    W: int = 32
    C: int = 3
    LABELS: List[str] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/cifar10_competition.npz"

    def __init__(self):
        data = self._load_data(self._URL)
        self.mu = 0
        self.std = 0

        for subset in ["train", "dev", "test"]:
            samples = torch.tensor(data[f"{subset}_images"], dtype=torch.float32)
            labels = torch.tensor(data[f"{subset}_labels"], dtype=torch.long)

            if subset == "train":
                self.mu = torch.mean(samples)
                self.std = torch.std(samples)

            setattr(self, subset, CIFAR10.TorchDataset(samples, labels, self.mu, self.std))


    def _load_data(self, path):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading CIFAR-10 dataset...", file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)
        return np.load(path)

    class TorchDataset(Dataset):

        def __len__(self):
            return self.labels.shape[0]

        def __init__(self, samples, labels, mu, std):

            self.mu = mu
            self.std = std

            self.transforms = [
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
            ]
            for transform in self.transforms:
                samples = torch.cat((samples,transform(samples)), axis=0)
                labels = torch.cat((labels,labels), axis=0)
            
            self.samples = (samples - self.mu) / self.std
            self.labels = labels


        def __getitem__(self, idx):
            return self.samples[idx].permute(2, 0, 1), self._smooth_labels(self.labels[idx])

        def _smooth_labels(self, labels, alpha:float = 0.1):
            labels = F.relu(F.one_hot(labels, 10) - alpha)
            labels += alpha/10
            return labels.view(-1)

    train: TorchDataset
    train: TorchDataset
    train: TorchDataset


    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: TorchDataset, predictions: Sequence[int]) -> float:
        gold = gold_dataset.labels

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: TorchDataset, predictions_file: TextIO) -> float:
        predictions = [int(line) for line in predictions_file]
        return CIFAR10.evaluate(gold_dataset, predictions)
