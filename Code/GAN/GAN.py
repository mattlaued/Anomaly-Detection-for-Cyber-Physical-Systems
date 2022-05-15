from tqdm import tqdm, tqdm_notebook
import tensorflow as tf
from tensorflow import keras
from keras import metrics, losses
# from keras.models import Model
from tensorflow_addons.metrics import F1Score
import numpy as np
from Data import getAttackDataIterator, getNormalDataIterator
from Code.GAN.Generator import Generator, Generator2
from Code.GAN.Discriminator import Discriminator, Discriminator2
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
        self.generatedDivergenceLoss = losses.KLDivergence()

    def compile(self,
                discOpt=keras.optimizers.Adam(1e-4),
                genOpt1=keras.optimizers.Adam(1e-4),
                genOpt2=keras.optimizers.Adam(1e-4),
                loss=keras.losses.BinaryFocalCrossentropy(),
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
        self.discOptimizer, self.genOptimizer1, self.genOptimizer2, self.loss = discOpt, genOpt1, genOpt2, loss


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
        with tf.GradientTape() as gTape1, tf.GradientTape() as gTape2, tf.GradientTape() as dTape:
            generated = self.generator(noise, training=True)

            # The distance of the generated data from the original data
            generatedDist = 1 / self.generatedDivergenceLoss(data, generated)#metrics.kullback_leibler_divergence(data, generated).min()

            realOut = self.discriminator(data, training=True)
            fakeOut = self.discriminator(generated, training=True)



            genLoss = self.compiled_loss(ones, fakeOut)
            discLoss = self.compiled_loss(zeros, fakeOut, zerosSampleWeights) + self.compiled_loss(ones, realOut,
                                                                                                   onesSampleWeights)

        genGrad1 = gTape1.gradient(genLoss, self.generator.trainable_variables)
        genGrad2 = gTape2.gradient(generatedDist, self.generator.trainable_variables)
        discGrad = dTape.gradient(discLoss, self.discriminator.trainable_variables)
        self.genOptimizer1.apply_gradients(zip(genGrad1, self.generator.trainable_variables))
        self.genOptimizer2.apply_gradients(zip(genGrad2, self.generator.trainable_variables))
        self.discOptimizer.apply_gradients(zip(discGrad, self.discriminator.trainable_variables))
        return {"disc loss": discLoss, "gen fooling loss": genLoss, 'gen noise loss': generatedDist}

    @tf.function
    def test_step(self, data):
        dat, labels = data
        pred = self.discriminator(dat)
        self.compiled_loss(labels, pred)
        self.compiled_metrics.update_state(labels, pred)
        result = {met.name: met.result() for met in self.metrics}
        return result


def loadGenerator(genEpoch, useGenerator2: bool):
    if genEpoch > 0:
        try:
            if useGenerator2:
                generator = keras.models.load_model(f'../../Checkpoints/GAN_generator2_epoch{genEpoch}')
            else:
                generator = keras.models.load_model(f'../../Checkpoints/GAN_generator_epoch{genEpoch}')
        except:
            if useGenerator2:
                print("Unable to load Generator2. Loading untrained Generator2")
                generator = Generator2()
            else:
                print("Unable to load Generator. Loading untrained Generator")
                generator = Generator()
    else:
        generator = Generator() if not useGenerator2 else Generator2()
    return generator


def loadGan(discEpoch, genEpoch, useDiscriminator2: bool, useGenerator2: bool, discLearningRate=1e-4, genfoolingLearningRate=1e-4, genNoiseLearningRate=1e-4):
    if discEpoch > 0:
        try:
            if useDiscriminator2:
                disc = keras.models.load_model(f'../../Checkpoints/GAN_discriminator2_epoch{discEpoch}')
            else:
                disc = keras.models.load_model(f'../../Checkpoints/GAN_discriminator_epoch{discEpoch}')
        except:
            if useDiscriminator2:
                print("Unable to load Discriminator2. Loading untrained Discriminator2")
                disc = Discriminator2()
            else:
                print("Unable to load Discriminator. Loading untrained Discriminator")
                disc = Discriminator()
    else:
        disc = Discriminator() if not useDiscriminator2 else Discriminator2()

    generator = loadGenerator(genEpoch, useGenerator2)
    generator.compile(jit_compile=True)

    disc.compile(jit_compile=True)

    gan = GAN(generator, disc)
    gan.compile(discOpt=keras.optimizers.Adam(discLearningRate),
                genOpt1=keras.optimizers.Adam(genfoolingLearningRate), genOpt2=keras.optimizers.Adam(genNoiseLearningRate))
    return gan

def loadTestDataAndTestLabels():
    attackIter = getAttackDataIterator(float('inf'), 5, True, True)
    testData, testLabels = list(zip(*list(attackIter)))
    testData, testLabels = np.ascontiguousarray(np.concatenate(testData)), np.ascontiguousarray(
        np.concatenate(testLabels))
    return testData, testLabels

def evalGan(discEpoch, genEpoch, useDiscriminator2: bool, useGenerator2: bool, testData=None, testLabels=None):
    gan = loadGan(discEpoch, genEpoch, useDiscriminator2, useGenerator2)
    [met.reset_states() for met in gan.metrics]
    if testData is None or testLabels is None:
        testData, testLabels = loadTestDataAndTestLabels()
    return gan.evaluate(testData, testLabels, return_dict=True, batch_size=8192), [testData, testLabels]


def saveGan(gan, discEpoch, genEpoch, useDiscriminator2: bool, useGenerator2: bool):
    if useDiscriminator2:
        gan.discriminator.save("../../Checkpoints/GAN_discriminator2_epoch{0}".format(discEpoch))
    else:
        gan.discriminator.save("../../Checkpoints/GAN_discriminator_epoch{0}".format(discEpoch))
    if useGenerator2:
        gan.generator.save("../../Checkpoints/GAN_generator2_epoch{0}".format(genEpoch))
    else:
        gan.generator.save("../../Checkpoints/GAN_generator_epoch{0}".format(genEpoch))


def continueTraining(discEpoch, genEpoch, stopEpoch, discLearningRate, genFoolingLearningRate, genNoiseLearningRate, batchSize, useDiscriminator2: bool, useGenerator2:bool, useScheduler: bool, isNotebook=False):
    """

    :param discEpoch: The training epoch to start training from (0 if brand new)
    :param genEpoch: The training epoch to start training from (0 if brand new)
    :param stopEpoch: The epoch to stop training at (float('inf')) if train forever
    :param discLearningRate:
    :param genFoolingLearningRate: The learning rate for the generator optimizer trying to fool the discriminator
    :param genNoiseLearningRate: The learning rate for the generator optimizer trying to have as much noise as possible
    :param isNotebook: Used for tqdm
    :return:
    """

    # If this crashes for you, the batch size may need to be lowered.
    sequenceLength = 5






    normalIter = getNormalDataIterator(8192, sequenceLength, True, False)

    normalIter._len = 61
    print(f"Training Batch Size: {batchSize}")
    testData, testLabels = None, None
    executor = ThreadPoolExecutor()
    testDataLabelFuture = executor.submit(loadTestDataAndTestLabels)
    executor.shutdown(wait=False)
    while True:
        if useScheduler:
            discLr, genLr1, genLr2 = scheduler(discEpoch, discLearningRate), scheduler(genEpoch, genFoolingLearningRate), scheduler(genEpoch, genNoiseLearningRate)
            print(f"\nDisc Epoch: {discEpoch + 1}\t Gen Epoch: {genEpoch + 1}\tDisc LR: {discLr}\tGen Fooling LR: {genLr1}\tGen Noise LR: {genLr2}")
            gan = loadGan(discEpoch, genEpoch, useDiscriminator2, useGenerator2, discLr,
                          genLr1, genLr2)
        else:
            gan = loadGan(discEpoch, genEpoch, useDiscriminator2, useGenerator2, discLearningRate, genFoolingLearningRate, genNoiseLearningRate)
        discEpoch, genEpoch = discEpoch + 1, genEpoch + 1
        if discEpoch > stopEpoch or genEpoch > stopEpoch:
            break

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
                # ret = gan.train_step(chunk)
                if discEpoch != genEpoch:
                    iterator.set_description(
                        f"Disc Epoch {discEpoch}\t Gen Epoch: {genEpoch}\tdisc loss: {ret[0]}\tgen fooling loss: {ret[1]}\tgen noise loss: {ret[2]}")
                else:
                    iterator.set_description(f"Epoch {discEpoch}\tdisc loss: {ret[0]}\tgen fooling loss: {ret[1]}\tgen noise loss: {ret[2]}")
                # iterator.update(1)
        iterator.close()
        saveGan(gan, discEpoch, genEpoch, useDiscriminator2, useGenerator2)
        if testData is None or testLabels is None:
            testData, testLabels = testDataLabelFuture.result()
        ret, (testData, testLabels) = evalGan(discEpoch, genEpoch, useDiscriminator2, useGenerator2, testData, testLabels)
def scheduler(epoch, lr):
    if epoch < 5:
        return lr * 0.95 ** epoch
    lr = lr * 0.95 ** 5
    if epoch < 10:
        return lr * 0.9 ** (epoch - 5)
    lr = lr * 0.9 ** 5
    if epoch < 15:
        return lr * 0.85 ** (epoch - 10)
    lr = lr * 0.85 ** 5
    if epoch < 20:
        return lr * 0.8 ** (epoch - 15)
    lr = lr * 0.8 ** 5
    if epoch < 25:
        return lr * 0.75 ** (epoch - 20)
    lr = lr * 0.75 ** 5
    lr = lr * 0.6 ** (epoch - 25)
    return lr

if __name__ == '__main__':
    # evalGan(28, 28, True, False)
    continueTraining(28, 28, stopEpoch=float('inf'), discLearningRate=0, genFoolingLearningRate=3e-3, genNoiseLearningRate=1e-3, batchSize=1024, useDiscriminator2=False, useGenerator2=False, useScheduler=True)