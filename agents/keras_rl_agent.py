import numpy as np

# from keras import Input, models, layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras import Input

from keras import __version__

tf.keras.__version__ = __version__

from keras.src.saving import serialization_lib

serialization_lib.enable_unsafe_deserialization()

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Input((3, height, width, channels)))
    model.add(
        Convolution2D(
            32,
            (8, 8),
            strides=(4, 4),
            activation="relu",
        )
    )
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(actions, activation="linear"))

    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.2,
        nb_steps=10000,
    )
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        enable_dueling_network=True,
        dueling_type="avg",
        nb_actions=actions,
        nb_steps_warmup=10000,
    )

    return dqn
