from keras import layers, Model, activations
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

class RNN_Line(layers.Layer):
    def __init__(self):
        super(RNN_Line, self).__init__()
        self.rnn = layers.RNN(layers.LSTMCell(512))
        self.leakyRelu = layers.LeakyReLU()
        self.dropout = layers.Dropout(0.2)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='tanh')
        self.dropout2 = layers.Dropout(0.2)


    def call(self, inputs, *args, **kwargs):
        with tf.device("/GPU:0"):
            return self.dropout2(self.dense(self.dropout(self.leakyRelu(self.rnn(inputs)))))


class CNN_Line(layers.Layer):
    def __init__(self):
        super(CNN_Line, self).__init__()
        self.conv1 = layers.Conv1D(64, 5, 2, padding='causal')
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv1D(128, 5, 2, padding='causal')
        self.leakyRelu2 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()

    def call(self, inputs, *args, **kwargs):
        out = self.conv1(inputs)
        out = self.leakyRelu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        with tf.device("/GPU:0"):
            out = self.leakyRelu2(out)
            out = self.dropout2(out)
            out = self.flatten(out)
            return out


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnnLine = CNN_Line()
        self.rnnLine = RNN_Line()
        self.concat = layers.Concatenate()
        self.dense = layers.Dense(1)
        self.sig = layers.Activation(activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        outB = self.cnnLine(inputs)
        with tf.device("/GPU:0"):
            outA = self.rnnLine(inputs)
            out = self.sig(self.dense(self.concat([outA, outB])))
            return out
