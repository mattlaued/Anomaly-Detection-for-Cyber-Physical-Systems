from Code.GAN.GAN import GAN
from Code.GAN.Generator import Generator
from Code.GAN.Discriminator import Discriminator
from tqdm import tqdm
from Data import getAttackDataIterator
from Data import getNormalDataIterator
from collections import defaultdict
import numpy as np
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
    i = 423
    normalIter = getNormalDataIterator(8192, 5, True, False)
    normalLenDict = defaultdict(lambda: 61)
    normalLenDict[256] = 1941
    while True:
        gan = GAN(generator, disc)
        gan.compile(jit_compile=True)
        i += 1
        # trainBatchSize = trainBatchSizes[i]
        trueBatchSize = 256  # max(trainBatchSize >> (shift + i - 2), 256)
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