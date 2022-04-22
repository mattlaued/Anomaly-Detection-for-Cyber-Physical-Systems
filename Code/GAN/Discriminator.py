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
        self.maxPool1 = layers.MaxPool1D()
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
            out = self.maxPool1(out)
            out = self.conv2(out)
            out = self.leakyRelu2(out)
            out = self.dropout2(out)
        return out
        # with tf.device("/GPU:0"):

        #     out = self.dropout2(out)
        #     out = self.dense1(self.flatten(out))
        #     return out
# class RNN_Line(layers.Layer):
#     def __init__(self, hiddenDim):
#         super(RNN_Line, self).__init__()
#         self.rnn = layers.RNN(layers.LSTMCell(hiddenDim))
#         self.leakyRelu = layers.LeakyReLU()
#         self.dropout = layers.Dropout(0.05)
#         self.flatten = layers.Flatten()
#         self.dense1 = layers.Dense(hiddenDim)
#         self.leakyRelu2 = layers.Activation(activations.leaky_relu)
#         self.dropout2 = layers.Dropout(0.05)
#         self.dense2 = layers.Dense(hiddenDim)
#         self.leakyRelu3 = layers.Activation(activations.leaky_relu)
#
#     def call(self, inputs, *args, **kwargs):
#         return self.leakyRelu3(
#             self.dense2(
#                 self.dropout2(
#                     self.leakyRelu2(
#                         self.dense1(
#                             self.dropout(
#                                 self.leakyRelu(
#                                     self.rnn(inputs))))))))
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


        # self.denseCols = [DenseLayer(16, 0.2) for i in range(51)]
        # self.concat = layers.Concatenate()
        # self.rnn = RNN_Line(hiddenDim)
        self.conv = CNN_Line(256)
        self.flatten = layers.Flatten()
        self.dense1 = DenseLayer(1024, 0.05)
        self.reshape = layers.Reshape((32, 32))
        # self.conv2 = CNN_Line(128)
        # self.flatten2 = layers.Flatten()
        # self.dense2 = DenseLayer(256, 0.1)
        # self.reshape2 = layers.Reshape((16, 16))
        self.conv3 = CNN_Line(128)
        self.flatten3 = layers.Flatten()
        self.dense3 = layers.Dense(1)
        self.sig = layers.Activation(activations.sigmoid)


    def call(self, inputs, training=None, mask=None):
        out = self.conv(inputs)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.reshape(out)
        # out = self.conv2(out)
        # out = self.flatten2(out)
        # out = self.dense2(out)
        # out = self.reshape2(out)
        out = self.conv3(out)
        out = self.flatten3(out)
        out = self.dense3(out)
        out = self.sig(out)
        return out






















