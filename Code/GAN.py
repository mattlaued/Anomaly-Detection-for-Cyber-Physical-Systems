import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations

import time
from Data import getAttackDataIterator, getNormalDataIterator, SequencedDataIterator
from Code.GAN.Generator import Generator
from Code.GAN.Discriminator import Discriminator

def make_generator_model(sequenceLength):
    model = tf.keras.Sequential()
    # There are 51 attributes, not counting time stamps and the label
    model.add(layers.Conv1D(10, 5, 2))
    model.add(layers.Activation(activations.sigmoid))
    model.add(layers.Dense(1024))

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.3))

    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(256))

    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(sequenceLength * 51))
    model.add(layers.Reshape((sequenceLength, 51)))

    # model.add(layers.Softmax())

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.RNN(layers.LSTMCell(512)))
    # model.add(layers.Conv1D(10, 5, 2))
    model.add(layers.Dropout(0.05))
    model.add(layers.Activation(activations.sigmoid))
    model.add(layers.Dense(1024))
    # model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
    #                         input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(256))

    # model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminatorLoss(realOut, fakeOut, loss):
    return loss(tf.ones_like(realOut), realOut) + loss(tf.zeros_like(fakeOut), fakeOut)


def generatorLoss(fakeOut, loss):
    return loss(tf.ones_like(fakeOut), fakeOut)


class GAN(object):
    def __init__(self, genOutputSequenceLength, use_progressBar=True):
        self._useProgressBar = use_progressBar
        self.generator = Generator((genOutputSequenceLength, 51))
        self.discriminator = Discriminator()#make_discriminator_model()
        self.generator.compile()
        self.discriminator.compile()
        self.loss = keras.losses.BinaryFocalCrossentropy(from_logits=True)
        self.generatorOptimizer = keras.optimizers.Adam(1e-9)
        self.discriminatorOptimizer = keras.optimizers.Adam(1e-9)

    def train(self, epochs: int, data, trainGenerator=False, trainDescriminator=False, label=None):
        if not trainGenerator and not trainDescriminator:
            raise Exception("Both trainGenerator and trainDescriminator are false. Must at least train one.")
        if (trainDescriminator and not trainGenerator) or (trainGenerator and not trainDescriminator):
            if trainDescriminator:
                if label is None:
                    raise Exception("label cannot be none if only training discriminator")
                else:
                    self.train_disc(epochs, data, label)
            else:
                self.train_gen(epochs, data)
        else:
            for epoch in range(epochs):
                if isinstance(data, SequencedDataIterator):
                    data.reset()
                batchIter = tqdm(data) if self._useProgressBar else data
                totalGenLoss = 0
                totalDiscLoss = 0
                batchesCompleted = 0
                for batch in batchIter:
                    genLoss, discLoss = self.trainStep_both(batch, self.generator, self.discriminator, self.loss,
                                                            self.generatorOptimizer, self.discriminatorOptimizer)

                    totalGenLoss += genLoss
                    totalDiscLoss += discLoss
                    batchesCompleted += 1
                    if self._useProgressBar:
                        genLoss = round(float(genLoss), 8)
                        genAvg = round(float(totalGenLoss) / batchesCompleted, 8)
                        discLoss = round(float(discLoss), 8)
                        discAvg = round(float(totalDiscLoss) / batchesCompleted, 8)

                        batchIter.set_description(
                            f"Gen Loss: {genLoss}\tAvg Gen Loss: {genAvg}\tDisc Loss: {discLoss}\tAVG Disc Loss: {discAvg}")

    def train_disc(self, epochs: int, dataIter, labelIter):
        for epoch in range(epochs):
            if isinstance(dataIter, SequencedDataIterator):
                dataIter.reset()
            if isinstance(labelIter, SequencedDataIterator):
                labelIter.reset()
            batchesCompleted = 0
            totalLoss = 0
            batchLabelIter = tqdm(zip(dataIter, labelIter)) if self._useProgressBar else zip(dataIter, labelIter)
            for batch, label in batchLabelIter:
                discLoss = self.trainStep_disc(batch, label, self.discriminator, self.loss, self.discriminatorOptimizer)
                totalLoss += discLoss
                batchesCompleted += 1
                if self._useProgressBar:
                    discLoss = round(float(discLoss), 8)
                    discAvg = round(float(totalLoss) / batchesCompleted, 8)
                    batchLabelIter.set_description(f"Disc Loss: {discLoss}\tAVG Disc Loss: {discAvg}")

    def train_gen(self, epochs: int, dataIter):
        for epoch in range(epochs):
            if isinstance(dataIter, SequencedDataIterator):
                dataIter.reset()
            batchesCompleted = 0
            totalLoss = 0
            batchIter = tqdm(dataIter) if self._useProgressBar else dataIter
            for batch in batchIter:
                genLoss = self.trainStep_gen(batch, self.generator, self.discriminator, self.loss,
                                             self.generatorOptimizer)
                totalLoss += genLoss
                batchesCompleted += 1
                if self._useProgressBar:
                    genLoss = round(float(genLoss), 8)
                    genAvg = round(float(totalLoss) / batchesCompleted, 8)
                    batchIter.set_description(f"Gen Loss: {genLoss}\tAvg Gen Loss: {genAvg}\t")

    def test_disc(self, dataLabelIter):
        batchIter = dataLabelIter if not self._useProgressBar else tqdm(dataLabelIter)
        dataPointsTested = 0
        totalLoss = 0
        for batchData, batchLabel in batchIter:
            batchOut = self.discriminator(batchData)
            dataPointsTested += len(batchData)
            totalLoss += float(self.loss(batchLabel.reshape(batchOut.shape), batchOut))
            if self._useProgressBar:
                batchIter.set_description(f"Total Loss: {totalLoss}\tAverage Loss: {totalLoss / dataPointsTested}")
        if dataPointsTested == 0:
            return totalLoss, 0
        return totalLoss, totalLoss / dataPointsTested

    @staticmethod
    @tf.function
    def generatorLoss(loss, fakeOutput):
        genLoss = loss(tf.ones_like(fakeOutput), fakeOutput)
        return genLoss

    @staticmethod
    @tf.function
    def discriminatorLoss(loss, realOutput, fakeOutput):
        realLoss = loss(tf.ones_like(realOutput), realOutput)
        fakeLoss = loss(tf.zeros_like(fakeOutput), fakeOutput)
        discLoss = realLoss + fakeLoss
        return discLoss

    @staticmethod
    # @tf.function
    def trainStep_both(batch, gen, disc, loss, genOpt, discOpt):
        noise = tf.random.normal(batch.shape)
        with tf.GradientTape() as gTape, tf.GradientTape() as dTape:
            generated = gen(noise, training=True)
            del noise
            fakeOutput = disc(generated, training=True)

            realOutput = disc(batch, training=True)

            genLoss = GAN.generatorLoss(loss, fakeOutput)
            discLoss = GAN.discriminatorLoss(loss, realOutput, fakeOutput)
        # Get Gradients
        genGradients = gTape.gradient(genLoss, gen.trainable_variables)
        discGradients = dTape.gradient(discLoss, disc.trainable_variables)
        # Apply Optimizer
        genOpt.apply_gradients(zip(genGradients, gen.trainable_variables))
        discOpt.apply_gradients(zip(discGradients, disc.trainable_variables))
        return genLoss, discLoss

    @staticmethod
    @tf.function
    def trainStep_gen(batch, gen, disc, loss, genOpt):
        noise = tf.random.normal(batch.shape)
        with tf.GradientTape() as tape:
            generated = gen(noise, training=True)
            fakeOutput = disc(generated, training=True)

            genLoss = GAN.generatorLoss(loss, fakeOutput)

        # Get Gradients
        genGradients = tape.gradient(genLoss, gen.trainable_variables)
        # Apply Optimizer
        genOpt.apply_gradients(zip(genGradients, gen.trainable_variables))
        return genLoss

    @staticmethod
    @tf.function
    def trainStep_disc(batchTrain, batchLabel, disc, loss, discOpt):
        with tf.GradientTape() as tape:
            batchOut = disc(batchTrain)
            batchLoss = loss(batchLabel, batchOut)
        discGrad = tape.gradient(batchLoss, disc.trainable_variables)
        discOpt.apply_gradients(zip(discGrad, disc.trainable_variables))
        return batchLoss


if __name__ == '__main__':
    # If this crashes for you, the batch size may need to be lowered.
    sequenceLength = 5
    testBatchSize = 16384  # Only effects how much data is loaded into memory at a time. Higher values does not

    print("Data points per testing batch: {0}".format(testBatchSize * sequenceLength * 51))

    attackIter = getAttackDataIterator(testBatchSize, sequenceLength, True, True)
    gan = GAN(sequenceLength)
    # gan.train(epochs=1, data=attackDatIter, label=attackLabelIter, trainDescriminator=True)
    # gan.train(epochs=1, data=normal, trainGenerator=True)
    best_avg_discLoss = float('inf')
    # EPOCHS = 200
    i = 0
    trainBatchSize = 128 #int(min(max(2 ** (14 - i), 128), 12000))
    while True:
        i += 1
        # trainBatchSize = int(min(max(2 ** (EPOCHS - i + 3), 1), 8192))

        print("Epoch {1}, Training Batch Size: {0}".format(trainBatchSize, i))
        print("Data points per training batch: {0}".format(trainBatchSize * sequenceLength * 51))
        normalIter = getNormalDataIterator(trainBatchSize, sequenceLength, True)
        gan.train(epochs=1, data=normalIter, trainDescriminator=True, trainGenerator=True)
        totalLoss, averageLoss = gan.test_disc(attackIter)
        if averageLoss < best_avg_discLoss:
            best_avg_discLoss = averageLoss
            gan.discriminator.save_weights(
                "../Checkpoints/GAN_discriminator_epoch{0}_avg_loss_{1}.ckpt".format(i, averageLoss))
            gan.generator.save_weights("../Checkpoints/GAN_generator_epoch{0}.ckpt".format(i))

