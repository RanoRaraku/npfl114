#!/usr/bin/env python3
import argparse
from typing import Tuple
import scipy
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float, float]:

    # Load data dist
    data_count = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")

            if line in data_count.keys():
                data_count[line] += 1
            else:
                 data_count[line] = 1


    # Load model PDF
    model_dist = {}
    with open(args.model_path, "r") as model:
        for line in model:
            k, v = line.rstrip("\n").split()
            model_dist[k] = float(v)

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_np = np.zeros(shape=(3,1), dtype=np.float32)
    model_np = np.zeros(shape=(3,1), dtype=np.float32)
    
    for idx, k in enumerate(data_count.keys()):
        data_np[idx, 0] = data_count[k]
        model_np[idx, 0] = model_dist[k] 
    data_np = data_np / data_np.sum()

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_np * np.log(data_np))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = -np.sum(data_np * np.log(model_np))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = np.sum(data_np * np.log(data_np/model_np))

    # Plati: entropy + KLD = cross_entropy
    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
