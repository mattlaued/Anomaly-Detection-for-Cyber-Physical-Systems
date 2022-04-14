import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, Model, models
import numpy as np
# Integer Columns: 2-4, 9-15, 19-24, 29-33, 42-43, 48-50
INTEGER_COLUMNS = [2, 3, 4] + list(range(9, 16)) + list(range(19, 25)) + list(range(29, 34)) + [42, 43] + list(range(48, 51))
REAL_COLUMNS = [i for i in range(51) if i not in INTEGER_COLUMNS]

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn = layers.RNN(layers.LSTMCell(720))
        self.dropout1 = layers.Dropout(0.1)
        self.sig1 = layers.Activation(activations.sigmoid)
        self.dense1 = layers.Dense(1024)
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.05)
        self.dense2 = layers.Dense(256)
        self.leakyRelu2 = layers.LeakyReLU()
        self.flatten = layers.Flatten()
        self.dense3 = layers.Dense(1)


    def call(self, inputs, training=None, mask=None):
        out = self.rnn(inputs)
        out = self.dropout1(out)
        out = self.sig1(out)
        out = self.dense1(out)
        out = self.leakyRelu1(out)
        out = self.dropout2(out)
        out = self.dense2(out)
        out = self.leakyRelu2(out)
        out = self.flatten(out)
        out = self.dense3(out)
        return out
