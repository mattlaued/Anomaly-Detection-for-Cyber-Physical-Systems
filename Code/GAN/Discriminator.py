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
        self.Arnn = layers.RNN(layers.LSTMCell(256))
        self.ALeakyRelu1 = layers.Activation(activations.leaky_relu)
        self.Adropout1 = layers.Dropout(0.2)
        self.ABatchNorm1 = layers.BatchNormalization()


        self.Bconv1 = layers.Conv1D(128, 1, padding='causal')
        self.BleakyRelu1 = layers.Activation(activations.leaky_relu)
        self.Bdropout1 = layers.Dropout(0.05)


        # self.Bconv2 = layers.Conv1D(10, 2)

        self.Bflatten1 = layers.Flatten()
        self.Bdense1 = layers.Dense(256)
        self.BleakyRelu2 = layers.Activation(activations.leaky_relu)
        self.Bdropout2 = layers.Dropout(0.2)
        self.BbatchNorm1 = layers.BatchNormalization()


        self.concat = layers.Concatenate()

        # layers.ConvLSTM2D(10, (2, 2), s)

        self.batchNorm1 = layers.BatchNormalization()
        self.dense1 = layers.Dense(1024)
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(1024)
        self.leakyRelu2 = layers.LeakyReLU()
        self.dropout3 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.dense3 = layers.Dense(1)
        self.sig2 = layers.Activation(activations.sigmoid)



    def call(self, inputs, training=None, mask=None):
        out1 = self.Adropout1(self.ALeakyRelu1(self.Arnn(inputs)))
        out1 = self.ABatchNorm1(out1)

        # out2 = self.BleakyRelu2(self.Bconv2(self.))
        out2 = self.Bconv1(inputs)
        out2 = self.BleakyRelu1(out2)
        out2 = self.Bdropout1(out2)
        out2 = self.Bflatten1(out2)
        out2 = self.Bdense1(out2)
        out2 = self.BleakyRelu2(out2)
        out2 = self.Bdropout2(out2)
        out2 = self.BbatchNorm1(out2)

        out = self.concat([out1, out2])

        out = self.batchNorm1(out)
        out = self.dense1(out)
        out = self.leakyRelu1(out)
        out = self.dropout2(out)
        out = self.dense2(out)
        out = self.leakyRelu2(out)
        out = self.dropout3(out)
        out = self.flatten(out)
        out = self.dense3(out)
        out = self.sig2(out)
        return out
