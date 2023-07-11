#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import tensorflow as tf
from tensorflow import keras

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=1024 , type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=7, type=int, help="Window size to use.")
parser.add_argument("--hidden_size", default=2048, type=int, help="Hidden layer size.")
parser.add_argument("--label_smoothing", default=0.3, type=float, help="Label smoothing")
parser.add_argument("--lr_decay", default=False, type=bool, help="Learning Rate decay.")
parser.add_argument("--dropout",default=0.1, type=float, help="Dropout rate")


# Functional API
class CharModel(tf.keras.Model):
    def __init__(self, args) -> None:
        inputs = keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
        hidden = keras.layers.Lambda(lambda x: tf.one_hot(x, args.alphabet_size + 2))(inputs)
        # hidden = keras.layers.Conv1D(
        #     filters=2*args.window+1,
        #     kernel_size=args.alphabet_size + 2,
        #     data_format="channels_first",
        #     groups=2*args.window+1,
        # )(hidden)     # takes too long to train
        hidden = keras.layers.Flatten()(hidden)
        hidden = keras.layers.Dense(args.hidden_size, activation=tf.nn.relu)(hidden)
        hidden = keras.layers.Dropout(args.dropout)(hidden)
        outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
        super().__init__(inputs=inputs, outputs=outputs)


        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=6000,
            decay_rate=0.96,
            staircase=True
        )
        self.compile(
            optimizer=tf.optimizers.AdamW(learning_rate=lr_schedule, jit_compile=False),
            loss=tf.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[tf.metrics.BinaryAccuracy(name="accuracy")],
        )
        self.tb_callback = keras.callbacks.TensorBoard(args.logdir)



def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
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

    # Load data
    dset = UppercaseData(args.window, args.alphabet_size)

    # Train model
    model = CharModel(args)
    model.fit(
        x=dset.train.data["windows"],
        y=dset.train.data["labels"],
        validation_data=(dset.dev.data["windows"],dset.dev.data["labels"]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[model.tb_callback],
    )

    # Generate correctly capitalized test set.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        pred = model.predict(dset.dev.data["windows"])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
