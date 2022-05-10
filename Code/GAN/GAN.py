from tqdm import tqdm, tqdm_notebook
import tensorflow as tf
from tensorflow import keras
from keras import metrics
# from keras.models import Model
from tensorflow_addons.metrics import F1Score
import numpy as np
from Data import getAttackDataIterator, getNormalDataIterator
from Code.GAN.Generator import Generator, GeneratorRepeater
from Code.GAN.Discriminator import Discriminator
import absl.logging
from concurrent.futures import ThreadPoolExecutor
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

GAN_CLASS_WEIGHTS = {0: 1, 1: 1}#{0: 1, 1: 12.7397503}


class GAN(keras.Model):
    def __init__(self, generator, discriminator, classWeights=None):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.classWeights = classWeights if classWeights is not None else GAN_CLASS_WEIGHTS

    def compile(self,
                discOpt=keras.optimizers.Adam(1e-4),
                genOpt=keras.optimizers.Adam(1e-4),
                loss=keras.losses.BinaryFocalCrossentropy(),
                # metrics=[metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(),
                #          metrics.FalseNegatives()],
                metrics=[F1Score(1, name="F1"), metrics.Accuracy(), metrics.Precision(), metrics.Recall(),
                         metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(),
                         metrics.FalseNegatives()],
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                jit_compile=None,
                **kwargs):
        super(GAN, self).compile(loss=loss, metrics=metrics, loss_weights=loss_weights,
                                 weighted_metrics=weighted_metrics,
                                 run_eagerly=run_eagerly, steps_per_execution=steps_per_execution,
                                 jit_compile=jit_compile, **kwargs)
        self.discOptimizer, self.genOptimizer, self.loss = discOpt, genOpt, loss

    def __call__(self, *args, **kwargs):
        return self.discriminator(*args, **kwargs)

    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        noise = tf.random.normal(data.shape)
        ones = tf.ones((data.shape[0],)) + 0.05 * tf.random.uniform((data.shape[0],))
        zeros = tf.zeros((data.shape[0],)) + 0.05 * tf.random.uniform((data.shape[0],))
        zerosSampleWeights = self.classWeights[0] * tf.ones_like(zeros)
        onesSampleWeights = self.classWeights[1] * tf.ones_like(ones)
        with tf.GradientTape() as gTape, tf.GradientTape() as dTape:
            generated = self.generator(noise, training=True)
            realOut = self.discriminator(data, training=True)
            fakeOut = self.discriminator(generated, training=True)

            genLoss = self.compiled_loss(ones, fakeOut)  # self.compiled_loss(ones, fakeOut, onesSampleWeights)
            discLoss = self.compiled_loss(zeros, fakeOut, zerosSampleWeights) + self.compiled_loss(ones, realOut,
                                                                                                   onesSampleWeights)
        # with tf.device('/CPU:0'):
        genGrad = gTape.gradient(genLoss, self.generator.trainable_variables)
        discGrad = dTape.gradient(discLoss, self.discriminator.trainable_variables)
        self.genOptimizer.apply_gradients(zip(genGrad, self.generator.trainable_variables))
        self.discOptimizer.apply_gradients(zip(discGrad, self.discriminator.trainable_variables))
        return {"disc loss": discLoss, "gen loss": genLoss}

    @tf.function
    def test_step(self, data):
        dat, labels = data
        pred = self.discriminator(dat)
        self.compiled_loss(labels, pred)
        self.compiled_metrics.update_state(labels, pred)
        result = {met.name: met.result() for met in self.metrics}
        return result


def loadGenerator(genEpoch):
    if genEpoch > 0:
        try:
            generator = keras.models.load_model(f'../../Checkpoints/GAN_generator_epoch{genEpoch}')
        except:
            print("Unable to load Generator. Loading untrained Generator")
            generator = Generator()
    else:
        generator = Generator()
    return generator


def loadGan(discEpoch, genEpoch, discLearningRate=1e-4, genLearningRate=1e-4):
    if discEpoch > 0:
        try:
            disc = keras.models.load_model(f'../../Checkpoints/GAN_discriminator_epoch{discEpoch}')
        except:
            print("Unable to load Discriminator. Loading untrained Discriminator")
            disc = Discriminator()
    else:
        disc = Discriminator()

    generator = loadGenerator(genEpoch)
    generator.compile(jit_compile=True)

    disc.compile(jit_compile=True)

    gan = GAN(generator, disc)
    gan.compile(discOpt=keras.optimizers.Adam(discLearningRate),
                genOpt=keras.optimizers.Adam(genLearningRate))
    return gan

def loadTestDataAndTestLabels():
    attackIter = getAttackDataIterator(float('inf'), 5, True, True)
    testData, testLabels = list(zip(*list(attackIter)))
    testData, testLabels = np.ascontiguousarray(np.concatenate(testData)), np.ascontiguousarray(
        np.concatenate(testLabels))
    return testData, testLabels

def evalGan(discEpoch, genEpoch, testData=None, testLabels=None):
    gan = loadGan(discEpoch, genEpoch)
    [met.reset_states() for met in gan.metrics]
    if testData is None or testLabels is None:
        testData, testLabels = loadTestDataAndTestLabels()
    return gan.evaluate(testData, testLabels, return_dict=True, batch_size=8192), [testData, testLabels]


def saveGan(gan, discEpoch, genEpoch):
    gan.discriminator.save("../../Checkpoints/GAN_discriminator_epoch{0}".format(discEpoch))
    gan.generator.save("../../Checkpoints/GAN_generator_epoch{0}".format(genEpoch))


def continueTraining(discEpoch, genEpoch, stopEpoch, discLearningRate, genLearningRate, batchSize, isNotebook=False, ):
    """

    :param discEpoch: The training epoch to start training from (0 if brand new)
    :param genEpoch: The training epoch to start training from (0 if brand new)
    :param stopEpoch: The epoch to stop training at (float('inf')) if train forever
    :param discLearningRate:
    :param genLearningRate:
    :param isNotebook: Used for tqdm
    :return:
    """

    # If this crashes for you, the batch size may need to be lowered.
    sequenceLength = 5






    normalIter = getNormalDataIterator(8192, sequenceLength, True, False)

    normalIter._len = 61  # 1941 if batchSize == 256 else 3882
    print(f"Training Batch Size: {batchSize}")
    testData, testLabels = None, None
    executor = ThreadPoolExecutor()
    testDataLabelFuture = executor.submit(loadTestDataAndTestLabels)
    executor.shutdown(wait=False)
    while True:
        gan = loadGan(discEpoch, genEpoch, discLearningRate, genLearningRate)
        discEpoch, genEpoch = discEpoch + 1, genEpoch + 1
        if discEpoch > stopEpoch or genEpoch > stopEpoch:
            break

        # print("\nEpoch {1}, Training Batch Size: {0}".format(batchSize, i))
        # print("Data points per training batch: {0}".format(batchSize * sequenceLength * 51))
        if isNotebook:
            iterator = tqdm_notebook(normalIter)
        else:
            iterator = tqdm(normalIter)
        for batch in iterator:
            batch = np.array(batch)
            np.random.shuffle(batch)
            chunks = np.array_split(batch, len(batch) // batchSize)
            for chunk in chunks:
                if chunk.shape != (batchSize, sequenceLength, 51):
                    continue
                ret = gan.train_on_batch(chunk, reset_metrics=True)
                if discEpoch != genEpoch:
                    iterator.set_description(
                        f"Disc Epoch {discEpoch}\t Gen Epoch: {genEpoch}\tdisc loss: {ret[0]}\tgen loss: {ret[1]}")
                else:
                    iterator.set_description(f"Epoch {discEpoch}\tdisc loss: {ret[0]}\tgen loss: {ret[1]}")
                # iterator.update(1)
        iterator.close()
        saveGan(gan, discEpoch, genEpoch)
        if testData is None or testLabels is None:
            testData, testLabels = testDataLabelFuture.result()
        ret, (testData, testLabels) = evalGan(discEpoch, genEpoch, testData, testLabels)


if __name__ == '__main__':
    continueTraining(28, 28, stopEpoch=float('inf'), discLearningRate=1e-5, genLearningRate=1e-5, batchSize=256)
