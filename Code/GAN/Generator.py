from keras import layers, activations, Model, models
import numpy as np
import tensorflow as tf


class DenseLayer(layers.Layer):
    def __init__(self, outputDim):
        super(DenseLayer, self).__init__()
        # self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(outputDim, use_bias=False)
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.Activation(activations.leaky_relu)

    def call(self, inputs, *args, **kwargs):
        out = self.dense1(inputs)
        out = self.batchNorm1(out)
        out = self.leakyrelu1(out)
        return out


class ConvPart(layers.Layer):
    def __init__(self, filters, size):
        super(ConvPart, self).__init__()
        self.conv1 = layers.Conv1DTranspose(filters, size, 1, padding='same', use_bias=False)
        self.batchNorm2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.Activation(activations.leaky_relu)
        self.conv2 = layers.Conv1DTranspose(filters, size, 1, padding='same', use_bias=False)
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.Activation(activations.leaky_relu)

    def call(self, inputs, *args, **kwargs):
        out = self.conv1(inputs)
        out = self.batchNorm2(out)
        out = self.leakyrelu2(out)
        out = self.conv2(out)
        out = self.batchNorm3(out)
        out = self.leakyrelu3(out)
        return out


class ColLine(layers.Layer):
    def __init__(self):
        super(ColLine, self).__init__()
        self.dense1 = DenseLayer(128)
        self.dense2 = DenseLayer(64)
        self.dense3 = layers.Dense(5, activation='tanh')

    def call(self, inputs, *args, **kwargs):
        out = self.dense1(inputs)
        out = self.dense2(out)
        out = self.dense3(out)
        return out


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.flatten = layers.Flatten()
        self.dense1 = DenseLayer(256)
        self.reshape1 = layers.Reshape((16, 16))
        self.conv = ConvPart(256, 6)
        self.flatten2 = layers.Flatten()
        self.dense2 = DenseLayer(256)
        self.reshape2 = layers.Reshape((16, 16))
        self.flatten3 = layers.Flatten()
        self.dense3 = DenseLayer(255)
        self.reshape3 = layers.Reshape((5, 51))
        self.conv2 = ConvPart(256, 4)
        self.conv3 = ConvPart(256, 2)
        self.convFinal = layers.Conv1DTranspose(51, 5, 1, padding='same', use_bias=False, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        out = self.flatten(inputs)
        out = self.dense1(out)
        out = self.reshape1(out)
        out = self.conv(out)
        out = self.flatten2(out)
        out = self.dense2(out)
        out = self.reshape2(out)
        out = self.conv2(out)
        out = self.flatten3(out)
        out = self.dense3(out)
        out = self.reshape3(out)
        out = self.conv3(out)
        out = self.convFinal(out)
        return out


class Generator2(Model):
    def __init__(self):
        super(Generator2, self).__init__()

        self.flatten = layers.Flatten()
        self.dense1 = DenseLayer(255)
        self.reshape1 = layers.Reshape((5, 51))
        self.conv = ConvPart(5, 51)
        self.conv2 = ConvPart(5, 51)
        self.conv3 = ConvPart(5, 51)
        self.convFinal = layers.Conv1DTranspose(51, 5, 1, padding='same', use_bias=False, activation='tanh')
    def call(self, inputs, training=None, mask=None):
        out = self.flatten(inputs)
        out = self.dense1(out)
        out = self.reshape1(out)
        out = self.conv(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.convFinal(out)
        return out

