import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers

import time


def make_generator_model():
    model = tf.keras.Sequential()
    # There are 51 attributes, not counting time stamps and the label
    model.add(layers.ConvLSTM1D(1024))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.3))

    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(256))

    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(2))
    model.add(layers.Softmax())

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(1000))

    # model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
    #                         input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256))


    # model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(2))

    return model
