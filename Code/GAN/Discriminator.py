from keras import layers, Model, activations, Sequential
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

    def call(self, inputs, *args, **kwargs):
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


class Discriminator2(Model):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.layer1 = layers.Permute((2, 1))
        self.layer2 = layers.Conv1D(64, 2, 2)
        self.layer3 = layers.LeakyReLU()
        self.layer4 = layers.Dropout(0.3)
        self.layer5 = layers.Conv1D(64, 2, 1)
        self.layer6 = layers.LeakyReLU()
        self.layer7 = layers.Dropout(0.3)
        self.layer8 = layers.Conv1D(64, 2, 2)
        self.layer9 = layers.LeakyReLU()
        self.layer10 = layers.Dropout(0.3)
        self.layer11 = layers.Conv1D(64, 2, 1)
        self.layer12 = layers.LeakyReLU()
        self.layer13 = layers.Dropout(0.3)
        self.layer14 = layers.Conv1D(64, 2, 2)
        self.layer15 = layers.LeakyReLU()
        self.layer16 = layers.Dropout(0.3)
        self.layer17 = layers.Conv1D(64, 2, 1)
        self.layer18 = layers.LeakyReLU()
        self.layer19 = layers.Dropout(0.3)
        self.avg, self.max = layers.AveragePooling1D(4, 1), layers.MaxPooling1D(4, 1)
        self.concat = layers.Concatenate()
        self.layer20 = layers.Flatten()
        self.layer21 = layers.Dense(1024, activation='tanh')
        self.layer22 = layers.Dropout(0.3)
        self.layer23 = layers.Reshape((32, 32))
        self.layer24 = layers.Conv1D(128, 5, 2)
        self.layer25 = layers.LeakyReLU()
        self.layer26 = layers.Dropout(0.3)
        self.layer27 = layers.Conv1D(128, 5, 2)
        self.layer28 = layers.LeakyReLU()
        self.layer29 = layers.Dropout(0.3)
        self.layer30 = layers.Conv1D(128, 2, 1)
        self.layer31 = layers.LeakyReLU()
        self.layer32 = layers.Dropout(0.3)
        self.layer33 = layers.Conv1D(128, 2, 1)
        self.layer34 = layers.LeakyReLU()
        self.layer35 = layers.Dropout(0.3)
        self.layer36 = layers.Conv1D(128, 2, 1)
        self.layer37 = layers.LeakyReLU()
        self.layer38 = layers.Dropout(0.3)
        self.layer39 = layers.Flatten()
        self.layer40 = layers.Dense(1)
        self.layer41 = layers.Activation(activations.sigmoid)


    def call(self, inputs, training=None, mask=None):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.layer19(out)
        avg, max = self.avg(out), self.max(out)
        out = self.concat([avg, max])
        out = self.layer20(out)
        out = self.layer21(out)
        out = self.layer22(out)
        out = self.layer23(out)
        out = self.layer24(out)
        out = self.layer25(out)
        out = self.layer26(out)
        out = self.layer27(out)
        out = self.layer28(out)
        out = self.layer29(out)
        out = self.layer30(out)
        out = self.layer31(out)
        out = self.layer32(out)
        out = self.layer33(out)
        out = self.layer34(out)
        out = self.layer35(out)
        out = self.layer36(out)
        out = self.layer37(out)
        out = self.layer38(out)
        out = self.layer39(out)
        out = self.layer40(out)
        out = self.layer41(out)
        return out
        # out = inputs
        # for layer in self.__layers:
        #     out = layer(out)
        # return out
