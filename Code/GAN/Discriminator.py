from keras import layers, Model, activations
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

#
class CNN_Line(layers.Layer):
    def __init__(self, filters: int):
        super(CNN_Line, self).__init__()
        self.conv1 = layers.Conv1D(filters, 5, 2, padding='same')
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.1)
        self.avgPool1 = layers.AveragePooling1D()
        self.conv2 = layers.Conv1D(filters, 2, 2, padding='same')
        self.leakyRelu2 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.1)
        # self.flatten = layers.Flatten()
        # self.dense1 = layers.Dense(128)

    def call(self, inputs, *args, **kwargs):
        with tf.device('/CPU:0'):
            out = self.conv1(inputs)
            out = self.leakyRelu1(out)
            out = self.dropout1(out)
            out = self.avgPool1(out)
            out = self.conv2(out)
            out = self.leakyRelu2(out)
            out = self.dropout2(out)
        return out

class DenseLayer(layers.Layer):
    def __init__(self, outputDim, dropout):
        super(DenseLayer, self).__init__()
        self.dense1 = layers.Dense(outputDim)
        self.batchNorm1 = layers.BatchNormalization()
        self.leakyrelu1 = layers.Activation(activations.leaky_relu)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, *args, **kwargs):
        out = self.dense1(inputs)
        out = self.batchNorm1(out)
        out = self.leakyrelu1(out)
        out = self.dropout(out)
        return out


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()


        self.concat = layers.Concatenate()
        self.denseLayer = DenseLayer(128, 0.2)
        self.flatten0 = layers.Flatten()
        self.denseLayer2 = DenseLayer(512, 0.2)
        self.denseLayer3 = DenseLayer(128, 0.2)
        self.conv = CNN_Line(256)
        self.flatten = layers.Flatten()
        self.dense1 = DenseLayer(1024, 0.05)
        self.reshape = layers.Reshape((32, 32))
        self.conv2 = CNN_Line(128)
        self.flatten2 = layers.Flatten()
        self.dense2 = DenseLayer(256, 0.1)
        self.reshape2 = layers.Reshape((16, 16))
        self.conv3 = CNN_Line(128)
        self.flatten3 = layers.Flatten()
        self.dense3 = layers.Dense(128)

        self.dense4 = layers.Dense(1)
        self.sig = layers.Activation(activations.sigmoid)


    def call(self, inputs, training=None, mask=None):
        avgs = tf.math.reduce_mean(inputs, -2)
        out1 = self.denseLayer(avgs, training=training, mask=mask)

        out2 = self.denseLayer2(self.flatten0(inputs), training=training, mask=mask)
        out2 = self.denseLayer3(out2, training=training, mask=mask)

        out3 = self.conv(inputs)
        out3 = self.flatten(out3)
        out3 = self.dense1(out3)
        out3 = self.reshape(out3)
        out3 = self.conv2(out3)
        out3 = self.flatten2(out3)
        out3 = self.dense2(out3)
        out3 = self.reshape2(out3)
        out3 = self.conv3(out3)
        out3 = self.flatten3(out3)
        out3 = self.dense3(out3)

        out = self.concat([out1, out2, out3])
        out = self.sig(self.dense4(out))
        return out










