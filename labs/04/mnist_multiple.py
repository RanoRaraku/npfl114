#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # TODO: The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature vector FV of each image.
        shared_conv = tf.keras.Sequential(
            layers = [
                tf.keras.layers.Conv2D(10,3,2,padding="valid", activation=tf.nn.relu),
                tf.keras.layers.Conv2D(10,3,2,padding="valid", activation=tf.nn.relu),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation=tf.nn.relu),
            ],
            name="shared_conv",
        )
        hidden_1 = shared_conv(images[0])
        hidden_2 = shared_conv(images[1])

        # TODO: Using the computed representations, the model should produce four outputs:
        # - first, compute _direct comparison_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations FV,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with `tf.nn.sigmoid` activation
        # - then, classify the computed representation FV of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation FV of the second image using
        #   the same layer (identical, i.e., with shared weights) into 10 classes;
        # - finally, compute _indirect comparison_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        direct_out = tf.keras.layers.Concatenate()([hidden_1, hidden_2])
        direct_out = tf.keras.layers.Dense(200, activation=tf.nn.relu)(direct_out)
        direct_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(direct_out)

        pred_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        pred_1 = pred_layer(hidden_1)
        pred_2 = pred_layer(hidden_2)

        indirect_out = tf.math.argmax(pred_1, axis=1) > tf.math.argmax(pred_2, axis=1)

        outputs = {
            "direct_comparison": direct_out,
            "digit_1": pred_1,
            "digit_2": pred_2,
            "indirect_comparison": indirect_out,
        }

        tf.losses.BinaryCrossentropy

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # TODO: Define the appropriate losses for the model outputs
        # "direct_comparison", "digit_1", "digit_2". Regarding metrics,
        # the accuracy of both the direct and indirect comparisons should be
        # computed; name both metrics "accuracy" (i.e., pass "accuracy" as the
        # first argument of the metric object).
        self.compile(
            optimizer=tf.keras.optimizers.Adam(jit_compile=False),
            loss={
                "direct_comparison": tf.losses.BinaryCrossentropy(),
                "digit_1": tf.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_comparison": tf.metrics.BinaryAccuracy(name="accuracy"),
                "indirect_comparison": tf.metrics.BinaryAccuracy(name="accuracy"),
            },
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        self.summary()

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
        self, mnist_dataset: MNIST.Dataset, args: argparse.Namespace, training: bool = False
    ) -> tf.data.Dataset:
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        # TODO: If `training`, shuffle the data with `buffer_size=10_000` and `seed=args.seed`.
        if training:
            dataset = dataset.shuffle(10000, seed=args.seed)

        # TODO: Combine pairs of examples by creating batches of size exactly 2 (you would throw
        # away the last example if the original dataset size were odd; but in MNIST it is even).
        dataset = dataset.batch(2)

        # TODO: Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys "digit_1", "digit_2", "direct_comparison",
        #   and "indirect_comparison".
        def create_element(images, labels):
            return (
                (images[0], images[1]),
                {
                    "digit_1":labels[0],
                    "digit_2":labels[1],
                    "direct_comparison": 1 if labels[0] > labels[1] else 0,
                    "indirect_comparison": 1 if labels[0] > labels[1] else 0,
                }
            )

        dataset = dataset.map(create_element)

        # TODO: Create batches of size `args.batch_size`
        dataset = dataset.batch(args.batch_size)

        return dataset


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model
    model = Model(args)

    # Construct suitable datasets from the MNIST data.
    train = model.create_dataset(mnist.train, args, training=True)
    dev = model.create_dataset(mnist.dev, args)

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
