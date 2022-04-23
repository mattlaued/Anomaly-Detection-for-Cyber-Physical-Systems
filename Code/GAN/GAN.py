from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import metrics, losses
# from keras.models import Model
from tensorflow_addons.metrics import F1Score
import numpy as np
from Data import getAttackDataIterator, getNormalDataIterator
from Code.GAN.Generator import Generator
from Code.GAN.Discriminator import Discriminator
from collections import defaultdict
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def discriminatorLoss(realOut, fakeOut, loss):
    return loss(tf.ones_like(realOut), realOut) + loss(tf.zeros_like(fakeOut), fakeOut)


def generatorLoss(fakeOut, loss):
    return loss(tf.ones_like(fakeOut), fakeOut)


class GAN(keras.Model):
    def __init__(self, generator, discriminator, use_progressBar=True):
        super(GAN, self).__init__()
        self._useProgressBar = use_progressBar
        self.generator = generator
        self.discriminator = discriminator

    def compile(self,
              discOpt=keras.optimizers.Adam(1e-4),
              genOpt=keras.optimizers.Adam(1e-4),
              loss=keras.losses.BinaryFocalCrossentropy(),
              metrics=[F1Score(1, name="F1"), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives(), metrics.BinaryAccuracy(), metrics.Accuracy()],
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              jit_compile=None,
              **kwargs):
        super(GAN, self).compile(loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics,
                                 run_eagerly=run_eagerly, steps_per_execution=steps_per_execution,
                                 jit_compile=jit_compile, **kwargs)
        self.discOptimizer, self.genOptimizer, self.loss = discOpt, genOpt, loss


    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        noise = tf.random.normal(data.shape)
        # fakeOut = self.generator(noise)
        fakeLables = tf.ones((data.shape[0],)) + 0.05 * tf.random.uniform((data.shape[0],))
        realLabels = tf.zeros((data.shape[0],)) + 0.05 * tf.random.uniform((data.shape[0],))

        # combined = tf.concat([fakeOut, data], axis=0)
        # labels = tf.concat([tf.ones((data.shape[0],)), tf.zeros((data.shape[0],))], axis=0)
        # labels += 0.05 * tf.random.uniform(labels.shape)
        misleadingLabels = tf.zeros((data.shape[0], 1))
        with tf.GradientTape() as gTape:
            fakeOut = self.generator(noise)
            pred = self.discriminator(fakeOut)

            genLoss = self.compiled_loss(misleadingLabels, pred)
        with tf.device("/CPU:0"):
            genGrad = gTape.gradient(genLoss, self.generator.trainable_weights)
        self.genOptimizer.apply_gradients(zip(genGrad, self.generator.trainable_weights))

        with tf.GradientTape() as dTape:
            predFake = self.discriminator(fakeOut, training=True)
            predReal = self.discriminator(data, training=True)
            discLoss = self.compiled_loss(fakeLables, predFake) + self.compiled_loss(realLabels, predReal)
        with tf.device('/CPU:0'):
            discGrad = dTape.gradient(discLoss, self.discriminator.trainable_weights)
        self.discOptimizer.apply_gradients(
            zip(discGrad, self.discriminator.trainable_weights)
        )


        # noise = tf.random.normal(data.shape)


        return {"disc loss": discLoss, "gen loss": genLoss}
    @tf.function
    def test_step(self, data):
        dat, labels = data
        labels = tf.convert_to_tensor(labels.reshape((len(labels), 1)))
        pred = self.discriminator(dat)
        self.compiled_loss(labels, pred)
        self.compiled_metrics.update_state(labels, pred)
        return {met.name: met.result() for met in self.metrics}


if __name__ == '__main__':
    # If this crashes for you, the batch size may need to be lowered.
    sequenceLength = 5
    testBatchSize = 8192  # Only effects how much data is loaded into memory at a time. Higher values does not

    print("\nData points per testing batch: {0}".format(testBatchSize * sequenceLength * 51))

    attackIter = getAttackDataIterator(testBatchSize, sequenceLength, True, True)
    disc = Discriminator()
    generator = Generator((sequenceLength, 51))
    disc.compile(jit_compile=True)
    generator.compile(jit_compile=True)
    disc.compile(jit_compile=True)
    gan = GAN(generator, disc)
    gan.compile()



    bestScore = 0
    # EPOCHS = 200

    # trainBatchSize = 128 #int(min(max(2 ** (14 - i), 128), 12000))
    trainBatchSizes = defaultdict(lambda: 512)
    sizes = [8192, 4096, 2048]
    for index in range(len(sizes)):
        start = (index * 10 + 1)
        for epoch in range(start, start + 11):
            trainBatchSizes[epoch] = sizes[index]
    # shift = 7
    i = 0
    normalIter = getNormalDataIterator(8192, 5, True, False)
    normalLenDict = defaultdict(lambda: 61)
    normalLenDict[256] = 1941
    while True:
        i += 1
        # trainBatchSize = trainBatchSizes[i]
        trueBatchSize = 256 # max(trainBatchSize >> (shift + i - 2), 256)
        print("\nEpoch {1}, Training Batch Size: {0}".format(trueBatchSize, i))
        print("Data points per training batch: {0}".format(trueBatchSize * sequenceLength * 51))
        normalIter._len = normalLenDict[trueBatchSize]
        iterator = tqdm(normalIter)
        for batch in iterator:
            realBatch = np.array(batch)
            np.random.shuffle(realBatch)
            batch = np.ascontiguousarray(realBatch)
            chunks = np.array_split(batch, len(batch) // trueBatchSize)
            for chunk in chunks:
                ret = gan.train_step(np.ascontiguousarray(chunk))
                iterator.set_description(str({key: str(float(ret[key].numpy())) for key in ret}))
                iterator.update(1)
        iterator.close()

        # if i % 5 == 0:
        attackIter._len = 372
        iterator = tqdm(attackIter)
        for testBatch in iterator:
            ret = gan.test_step(testBatch)
            string = str({key: str(float(ret[key].numpy())) for key in ret})
            items = string.split(", ")
            newString = ", ".join(items[:len(items) // 2]) + "\t" + ", ".join(items[len(items) // 2:])
            iterator.set_description(newString)
        if float(ret['accuracy']) > bestScore:
            bestScore = float(ret['accuracy'])
            gan.discriminator.save_weights(
                "../Checkpoints/GAN_discriminator_epoch{0}_accuracy_{1}.ckpt".format(i, bestScore))
            gan.generator.save_weights("../Checkpoints/GAN_generator_epoch{0}.ckpt".format(i))
        iterator.close()
