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
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):


    def __init__(self, args: argparse.Namespace) -> None:
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        ## 1 --cnn=F,H-100
        #flatten = tf.keras.layers.Flatten()(inputs)
        #hidden = tf.keras.layers.Dense(100, activation=tf.nn.relu)(flatten)
        
        ## 2 --cnn=F,H-100,D-0.5
        #flatten = tf.keras.layers.Flatten()(inputs)
        #hidden = tf.keras.layers.Dense(100, activation=tf.nn.relu)(flatten)
        #hidden = tf.keras.layers.Dropout(0.5)(hidden)

        ## 3 --cnn=M-5-2,F,H-50
        #hidden = tf.keras.layers.MaxPool2D((5,5),2)(inputs)
        #hidden = tf.keras.layers.Flatten()(hidden)
        #hidden = tf.keras.layers.Dense(50, activation=tf.nn.relu)(hidden)

        ## 4 --cnn=C-8-3-5-same,C-8-3-2-valid,F,H-50
        #hidden = tf.keras.layers.Conv2D(8,3,5,padding="same", activation=tf.nn.relu)(inputs)
        #hidden = tf.keras.layers.Conv2D(8,3,2,padding="valid", activation=tf.nn.relu)(hidden)
        #hidden = tf.keras.layers.Flatten()(hidden)
        #hidden = tf.keras.layers.Dense(50, activation=tf.nn.relu)(hidden)

        ## 5 --cnn=CB-6-3-5-valid,F,H-32
        #hidden = tf.keras.layers.Conv2D(6,3,5,use_bias=False)(inputs)
        #hidden = tf.keras.layers.BatchNormalization()(hidden)
        #hidden = tf.keras.layers.Flatten()(hidden)
        #hidden = tf.keras.layers.Dense(32, activation=tf.nn.relu)(hidden)        

        ## 6 --cnn=CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50
        hidden = tf.keras.layers.Conv2D(8,3,5,use_bias=False, padding="valid")(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        hidden_1 = tf.keras.layers.Conv2D(8,3,1,use_bias=False, padding="same")(hidden)
        hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
        hidden_1 = tf.keras.layers.ReLU()(hidden_1)
        hidden_1 = tf.keras.layers.Conv2D(8,3,1,use_bias=False, padding="same")(hidden_1)
        hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
        hidden_1 = tf.keras.layers.ReLU()(hidden_1)
        hidden_1 = tf.keras.layers.Add()([hidden_1, hidden])    # <- Residual

        ## Output
        hidden = tf.keras.layers.Flatten()(hidden_1)
        hidden = tf.keras.layers.Dense(50, activation=tf.nn.relu)(hidden) 
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


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

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
