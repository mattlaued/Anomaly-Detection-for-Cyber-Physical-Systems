import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, Model, models
import numpy as np
# Integer Columns: 2-4, 9-15, 19-24, 29-33, 42-43, 48-50
INTEGER_COLUMNS = [2, 3, 4] + list(range(9, 16)) + list(range(19, 25)) + list(range(29, 34)) + [42, 43] + list(range(48, 51))
REAL_COLUMNS = [i for i in range(51) if i not in INTEGER_COLUMNS]

class Generator(Model):
    def __init__(self, outputDim: tuple):
        super(Generator, self).__init__()
        self.conv1 = layers.Conv1D(10, 5, 2)
        self.sig1 = layers.Activation(activations.sigmoid)
        self.dense1= layers.Dense(1024)
        self. batchNorm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        self.leakyrelu1 = layers.LeakyReLU()
        self.flatten1 = layers.Flatten()
        self.dense2 = layers.Dense(256)
        self.batchNorm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)
        self.leakyrelu2 = layers.LeakyReLU()
        self.dense3 = layers.Dense(64)
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU()
        self.dense4 = layers.Dense(int(np.prod(outputDim)))
        self.reshape = layers.Reshape(outputDim)

    @staticmethod
    def formatOutput(output):
        breakpoint()

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.sig1(out)
        out = self.dense1(out)
        out = self.batchNorm1(out)
        out = self.dropout1(out)
        out = self.leakyrelu1(out)
        out = self.flatten1(out)
        out = self.dense2(out)
        out = self.batchNorm2(out)
        out = self.dropout2(out)
        out = self.leakyrelu2(out)
        out = self.dense3(out)
        out = self.batchNorm3(out)
        out = self.leakyrelu3(out)
        out = self.dense4(out)
        out = self.reshape(out)
        return out
