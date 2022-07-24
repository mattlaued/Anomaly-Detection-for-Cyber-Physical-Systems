import numpy as np
import xgboost as xgb
from numba import cuda
from math import ceil
class Model(object):
    def __init__(self, regressorFile="", classifierFile="", regKwargs=None, classifierKwargs=None):
        if regressorFile != "":
            self.regressor = xgb.XGBRegressor()
            self.regressor.load_model(regressorFile)
        else:
            self.regressor = xgb.XGBRegressor(**regKwargs)
        if classifierFile != "":
            self.classifier = xgb.XGBClassifier()
            self.classifier.load_model(classifierFile)
        else:
            # 'multi:softprob',
            self.classifier = xgb.XGBClassifier(**classifierKwargs)

    def predict(self, x, threshold):
        regOutput = self.regressor.predict(x)
        classOutput = self.classifier.predict_proba(regOutput)
        prediction = np.zeros((classOutput.shape[0]))
        calcPrediction[ceil(float(prediction.size) / 32.0), 32](classOutput, prediction, threshold)
        return prediction
@cuda.jit
def calcPrediction(probs, output, activationThreshold):
    # Thead id in a 1D Block
    tx = cuda.threadIdx.x
    # Block id in a 1D Grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute the flattened index inside the array
    pos = tx + ty * bw
    if pos < probs.shape[0]:
        entry = probs[pos]
        maximum = -1
        maxArg = -1
        for i in range(entry.shape[-1]):
            if entry[i] > maximum:
                maxArg = i
                maximum = entry[i]
        if maximum < activationThreshold:
            output[pos] = 1
