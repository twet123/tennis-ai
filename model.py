import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers


def create_model(actions):
    model = keras.Sequential()
    model.add(
        layers.Lambda(
            lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]),
            output_shape=(84, 84, 4),
            input_shape=(4, 84, 84),
        )
    )
    model.add(
        layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=(4, 84, 84))
    )
    model.add(layers.Conv2D(64, 4, strides=2, activation="relu"))
    model.add(layers.Conv2D(32, 8, strides=4, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(actions, activation="relu"))

    return model
