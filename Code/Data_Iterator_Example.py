import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers


import time
from Data import getAttackDataIterator, getNormalDataIterator
if __name__ == '__main__':
    batchSize, sequenceLength, numDatCols = 128, 20, 51
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(batchSize, sequenceLength * numDatCols)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile()
    attackDatIter = getAttackDataIterator(batchSize=batchSize, sequenceLength=sequenceLength, includeData=True)
    attackLabelIter = getAttackDataIterator(batchSize=batchSize,sequenceLength=sequenceLength, includeLabel=True)
    dat1 = attackDatIter.__next__()
    flattened = dat1.reshape(1, dat1.shape[0], int(np.prod(dat1.shape[1:])))
    out = model(flattened)
    print(out)

