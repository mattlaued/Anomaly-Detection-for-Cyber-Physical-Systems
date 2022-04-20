from keras import layers, Model



class LineA(layers.Layer):
    def __init__(self):
        super(LineA, self).__init__()
        self.rnn = layers.RNN(layers.LSTMCell(256))
        self.leakyRelu = layers.LeakyReLU()
        self.dropout = layers.Dropout(0.3)
        self.flatten = layers.Flatten()

    def call(self, inputs, *args, **kwargs):
        return self.dropout(self.leakyRelu(self.rnn(inputs)))

class LineB(layers.Layer):
    def __init__(self):
        super(LineB, self).__init__()
        self.conv1 = layers.Conv1D(64, 5, 2, padding='causal')
        self.leakyRelu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv1D(128, 5, 2, padding='causal')
        self.leakyRelu2 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()

    def call(self, inputs, *args, **kwargs):
        return self.flatten(
            self.dropout2(self.leakyRelu2(self.conv2(self.dropout1(self.leakyRelu1(self.conv1(inputs)))))))


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lineA = LineA()
        self.lineB = LineB()

        self.concat = layers.Concatenate()
        self.dense = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        out = self.dense(self.concat([self.lineA(inputs), self.lineB(inputs)]))
        return out
