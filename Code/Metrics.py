from numba import cuda
from math import ceil
import numpy as np
@cuda.jit
def cEvaluate(prediction, labels, truePositives, trueNegatives, falsePositives, falseNegatives):
    # Thead id in a 1D Block
    tx = cuda.threadIdx.x
    # Block id in a 1D Grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute the flattened index inside the array
    pos = tx + ty * bw
    if pos < prediction.size:
        pred = prediction[pos]
        label = labels[pos]
        truePositives[pos] = 1 if pred == 1 and label == 1 else 0
        trueNegatives[pos] = 1 if pred == 0 and label == 0 else 0
        falsePositives[pos] = 1 if pred == 1 and label == 0 else 0
        falseNegatives[pos] = 1 if pred == 0 and label == 1 else 0

def evaluateMetrics(prediction, labels):
    threadsPerBlock = 32
    blocksPerGrid = ceil(float(len(prediction)) / float(threadsPerBlock))

    truePositives = np.zeros((len(prediction),), dtype=int)
    trueNegatives = np.zeros((len(labels)), dtype=int)
    falsePositives = np.zeros((len(labels)), dtype=int)
    falseNegatives = np.zeros((len(prediction),), dtype=int)
    cEvaluate[blocksPerGrid, threadsPerBlock](prediction, labels, truePositives, trueNegatives, falsePositives, falseNegatives)
    tp = float(truePositives.sum())
    tn = float(trueNegatives.sum())
    fp = float(falsePositives.sum())
    fn = float(falseNegatives.sum())
    res = {}
    # res['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    # if tp + fp > 0:
    #     res['Specificity'] =  tn / (tn + fp)
    if tp + fp > 0:
        res['Precision'] = tp / (tp + fp)
    else:
        res['Precision'] = 0.0
    if tp + fn > 0:
        res['Recall'] = tp / (tp + fn)
    else:
        res['Recall'] = 0.0
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        res['F1'] = f1
    except:
        res['F1'] = 0.0
    res['True_Positives'] = tp
    res['True_Negatives'] = tn
    res['False_Positives'] = fp
    res['False_Negatives'] = fn

    return res




