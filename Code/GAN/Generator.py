from keras import layers, activations, Model, models
import numpy as np

# Integer Columns: 2-4, 9-15, 19-24, 29-33, 42-43, 48-50
INTEGER_COLUMNS = [2, 3, 4] + list(range(9, 16)) + list(range(19, 25)) + list(range(29, 34)) + [42, 43] + list(
    range(48, 51))
REAL_COLUMNS = [i for i in range(51) if i not in INTEGER_COLUMNS]


class DensePart(layers.Layer):
    def __init__(self, outputDim: tuple):
        super(DensePart, self).__init__()
        self.flatten1 = layers.Flatten()
        self.dense1 = layers.Dense(outputDim[0] * outputDim[1], use_bias=False, input_shape=outputDim)
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.Activation(activations.leaky_relu)
        self.reshape1 = layers.Reshape((outputDim[0], outputDim[1]))

    def call(self, inputs, *args, **kwargs):
        out = self.flatten1(inputs)
        out = self.dense1(out)
        out = self.batchNorm1(out)
        out = self.leakyrelu1(out)
        out = self.reshape1(out)
        return out


class ConvPart(layers.Layer):
    def __init__(self, outputDim: tuple):
        super(ConvPart, self).__init__()
        self.conv1 = layers.Conv1DTranspose(128, 5, 1, padding='same', use_bias=False)
        self.batchNorm2 = layers.BatchNormalization()
        self.leakyrelu2 = layers.Activation(activations.leaky_relu)
        self.conv2 = layers.Conv1DTranspose(64, 5, 1, padding='same', use_bias=False)
        self.batchNorm3 = layers.BatchNormalization()
        self.leakyrelu3 = layers.Activation(activations.leaky_relu)
        self.conv3 = layers.Conv1DTranspose(outputDim[1], 5, 1, padding='same', use_bias=False, activation='tanh')

    def call(self, inputs, *args, **kwargs):
        out = self.conv1(inputs)
        out = self.batchNorm2(out)
        out = self.leakyrelu2(out)
        out = self.conv2(out)
        out = self.batchNorm3(out)
        out = self.leakyrelu3(out)
        out = self.conv3(out)
        return out


class Generator(Model):
    def __init__(self, outputDim: tuple):
        super(Generator, self).__init__()
        self.densePart = DensePart((outputDim[0], 256))
        self.convPart = ConvPart(outputDim)

        # self.flatten1 = layers.Flatten()
        # self.dense1 = layers.Dense(outputDim[0] * 256, use_bias=False, input_shape=outputDim)
        # self.batchNorm1 = layers.BatchNormalization()
        # self.leakyrelu1 = layers.Activation(activations.leaky_relu)
        # self.reshape1 = layers.Reshape((outputDim[0], 256))
        # self.conv1 = layers.Conv1DTranspose(128, 5, 1, padding='same', use_bias=False)
        # self.batchNorm2 = layers.BatchNormalization()
        # self.leakyrelu2 = layers.Activation(activations.leaky_relu)
        #
        # self.conv2 = layers.Conv1DTranspose(64, 5, 1, padding='same', use_bias=False)
        # self.batchNorm3 = layers.BatchNormalization()
        # self.leakyrelu3 = layers.Activation(activations.leaky_relu)
        #
        # self.conv3 = layers.Conv1DTranspose(outputDim[1], 5, 1, padding='same', use_bias=False, activation='tanh')

    def call(self, inputs, training=None, mask=None):
        out = self.densePart(inputs)
        out = self.convPart(out)
        return out
