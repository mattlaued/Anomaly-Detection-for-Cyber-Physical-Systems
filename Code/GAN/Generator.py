import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1DTranspose
from keras import layers, activations, Model, models
import numpy as np
# Integer Columns: 2-4, 9-15, 19-24, 29-33, 42-43, 48-50
INTEGER_COLUMNS = [2, 3, 4] + list(range(9, 16)) + list(range(19, 25)) + list(range(29, 34)) + [42, 43] + list(range(48, 51))
REAL_COLUMNS = [i for i in range(51) if i not in INTEGER_COLUMNS]

class Generator(Model):
    def __init__(self, outputDim: tuple):
        super(Generator, self).__init__()
        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(5 * 51, use_bias=False)
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.Activation(activations.leaky_relu)
        self.reshape1 = layers.Reshape((5, 51))
        self.conv1 = layers.Conv1D(10, 5, 2)



        self.leakyrelu2 = layers.LeakyReLU()
        self.flatten2 = layers.Flatten()
        self.dense2 = layers.Dense(512, use_bias=False)
        self.batchNorm2 = layers.BatchNormalization()
        self.leakyrelu3 = layers.LeakyReLU()
        self.dense3 = layers.Dense(1024, use_bias=False)
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyrelu4 = layers.LeakyReLU()
        self.dense4 = layers.Dense(int(np.prod(outputDim)), use_bias=False)
        self.reshape = layers.Reshape(outputDim)
        # self.flatten1 = layers.Flatten()
        # self.dense1 = layers.Dense(7 * 128, use_bias=False)
        # self.batchNorm1 = layers.BatchNormalization()
        # self.act1 = layers.Activation(activations.leaky_relu)
        # self.reshape1 = layers.Reshape((7, 128))
        # self.convTranspose1 = Conv1DTranspose(64, 5, strides=1, padding='same', use_bias=False)
        # self.batchNorm2 = layers.BatchNormalization()
        # self.act2 = layers.Activation(activations.leaky_relu)
        # self.convTranspose2 = Conv1DTranspose(64, 5, 2, padding='causal', use_bias=False)

        #
        # self.conv1 = layers.Conv1D(10, 5, 1)
        # self.sig1 = layers.Activation(activations.sigmoid)
        # self.dense1= layers.Dense(256)
        # self. batchNorm1 = layers.BatchNormalization()
        # # self.dropout1 = layers.Dropout(0.2)
        # self.leakyrelu1 = layers.LeakyReLU()
        # self.flatten1 = layers.Flatten()
        # self.dense2 = layers.Dense(1024)
        # self.batchNorm2 = layers.BatchNormalization()
        # self.dropout2 = layers.Dropout(0.2)
        # self.leakyrelu2 = layers.LeakyReLU()
        # self.dense3 = layers.Dense(1024)
        # self.batchNorm3 = layers.BatchNormalization()
        # self.leakyrelu3 = layers.LeakyReLU()
        # self.dense4 = layers.Dense(int(np.prod(outputDim)))
        # self.reshape = layers.Reshape(outputDim)


    def call(self, inputs, training=None, mask=None):
        out = self.flatten1(inputs)
        out = self.dense1(out)
        out = self.batchNorm1(out)
        out = self.leakyrelu1(out)

        out = self.reshape1(out)
        out = self.conv1(out)

        out = self.leakyrelu2(out)
        out = self.flatten2(out)
        out = self.dense2(out)
        out = self.batchNorm2(out)
        out = self.leakyrelu3(out)
        out = self.dense3(out)
        out = self.batchNorm3(out)
        out = self.leakyrelu4(out)
        out = self.dense4(out)
        out = self.reshape(out)
        return out
