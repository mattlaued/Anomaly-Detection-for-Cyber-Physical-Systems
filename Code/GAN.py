import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers

import time
from Data import getAttackDataIterator, getNormalDataIterator

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
def discriminatorLoss(realOut, fakeOut):
    return cross_entropy(tf.ones_like(realOut), realOut) + cross_entropy(tf.zeros_like(fakeOut), fakeOut)
def generatorLoss(fakeOut):
    return cross_entropy(tf.ones_like(fakeOut), fakeOut)

@tf.function
def trainStep(batch):
    pass

# def train()
# class GAN_Trainer(object):
#     def __init__(self, generator, discriminator, ):

if __name__ == '__main__':

    normal = getNormalDataIterator(10, 5)
    attack = getAttackDataIterator(10, 5)
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    generatorOptimizer = keras.optimizers.Adam(1e-4)
    discriminatorOptimizer = keras.optimizers.Adam(1e-4)
    z = 3
