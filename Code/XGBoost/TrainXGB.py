# Ben Podrazhansky
# from Multioptimizer import MultiOptimizer
from Code.XGBoost.Model import Model
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import sqlite3
from Data import AttackDBPath, NormalDBPath
import pandas as pd
from scipy.stats import mode
from collections import defaultdict
from numba import cuda
from Code.F1Metrics import evaluateMetrics


dataCols = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'DPIT301',
            'LIT301', 'AIT402', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502',
            'FIT503', 'PIT501', 'PIT503', 'FIT601']

def evalData():
    con = sqlite3.connect(AttackDBPath())
    cursor = con.cursor()
    data = np.array(
        list(cursor.execute(f"SELECT Timestamp, ATTACK FROM Attack ORDER BY Timestamp")))
    data = sliding_window_view(data, (5, 2)).squeeze()
    # evalX = np.array(data[:, :, 1:-1], dtype=float)
    evalY = np.array(np.array(data[:, :, -1], dtype=float), dtype=int).max(-1)
    xDf = pd.read_sql("SELECT * FROM Attack", sqlite3.connect(AttackDBPath()), parse_dates=["Timestamp"])
    sensor_indices = 1 + np.array(
        [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47], dtype=int)
    x = np.array(xDf.iloc[:, sensor_indices])
    evalX = sliding_window_view(x, (5, x.shape[-1])).squeeze()
    evalX = np.array(evalX).reshape((evalX.shape[0], np.prod(evalX.shape[1:])))

    #evalX.reshape((evalX.shape[0], int(np.prod(evalX.shape[1:]))))
    return evalX, evalY


def regressorTrainData():
    normalDf = pd.read_sql("SELECT * FROM Normal", sqlite3.connect(NormalDBPath()), parse_dates=["Timestamp"])
    sensor_indices = 1 + np.array(
        [0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47],
        dtype=int)
    x = np.array(normalDf.iloc[:, sensor_indices])
    labels = np.array(pd.read_sql("SELECT * FROM RegressionLabels", sqlite3.connect("../../Data/Regression_Labels.db"),
                                  parse_dates=["Start_Time", "End_Time"]).drop(columns=["Start_Time", "End_Time"]))[:, :-1]

    train = sliding_window_view(x, (5, x.shape[-1])).squeeze()
    train = np.array(train).reshape((train.shape[0], np.prod(train.shape[1:])))
    return train, labels

@cuda.jit
def labelsToProbs(labels, labelProbs):
    # Thead id in a 1D Block
    tx = cuda.threadIdx.x
    # Block id in a 1D Grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute the flattened index inside the array
    pos = tx + ty * bw
    if pos < labels.size:  # check array boundaries
        labelProbs[pos][labels[pos]] = 1.0

def classifierTrainData():
    con = sqlite3.connect("../../Data/Regression_Labels.db")
    cursor = con.cursor()
    listOfRegressionLabels = np.array(list(cursor.execute("SELECT * FROM Regression_Labels ORDER BY Start_Time")))
    x = np.array(listOfRegressionLabels[:, 3:], dtype=float)
    con = sqlite3.connect("../../Data/KMeans_Predictions.db")
    cursor = con.cursor()
    kMeansL = np.array(list(cursor.execute("SELECT * FROM Preprocessed_Normal ORDER BY Timestamp")))
    sliding_window_view(kMeansL, kMeansL.shape)
    kMeansL = sliding_window_view(kMeansL, (5, 2)).squeeze()
    kMeansL = np.array(kMeansL[:, :, -1], dtype=int)
    kMeansL = mode(kMeansL, -1).mode

    y = kMeansL.squeeze()
    numClasses = np.unique(y).size
    # yProbs = np.zeros((y.shape[0], numClasses))
    # labelsToProbs[ceil(float(y.size) / 32.0), 32](y, yProbs)

    return x, y

def trainNewModel(regressorKwargs: dict, classifierKwargs: dict):
    model = Model(regKwargs=regressorKwargs, classifierKwargs=classifierKwargs)
    regX, regY = regressorTrainData()
    reg = model.regressor
    reg.fit(regX, regY)
    reg.save_model('currentRegressor.xgb')
    del regX, regY


    classX, classY = classifierTrainData()
    classifier = model.classifier
    classifier.fit(classX, classY)
    classifier.save_model('currentClassifier.xgb')
    del classX, classY
    # model.regressor.load_model('currentRegressor.xgb')
    # model.classifier.load_model('currentClassifier.xgb')
    return model
def evalModel(model):
    x, y = evalData()
    output = model.predict(x)

def trainAndEvaluate(kwargs):
    regKwargs = { 'colsample_bytree': kwargs['reg_colsample_bytree'], 'learning_rate': kwargs['reg_learning_rate'],
                 'max_depth': None if kwargs['reg_max_depth'] is None else int(kwargs['reg_max_depth']), 'alpha': kwargs['reg_alpha'],
                 'n_estimators': None if kwargs['reg_n_estimators'] is None else int(kwargs['reg_n_estimators']), 'scale_pos_weight': kwargs['reg_scale_pos_weight']
                 }
    for key in list(regKwargs.keys()):
        if regKwargs[key] is None:
            regKwargs.pop(key)
        elif isinstance(regKwargs[key], str):
            regKwargs[key] = float(regKwargs[key])
    classKwargs = { 'colsample_bytree': kwargs['class_colsample_bytree'], 'learning_rate': kwargs['class_learning_rate'],
                 'max_depth': int(kwargs['class_max_depth']), 'alpha': kwargs['class_alpha'],
                 'n_estimators': int(kwargs['class_n_estimators']), 'scale_pos_weight': kwargs['class_scale_pos_weight']
                 }
    for key in list(classKwargs.keys()):
        if classKwargs[key] is None:
            classKwargs.pop(key)
        elif isinstance(classKwargs[key], str):
            classKwargs[key] = float(classKwargs[key])

    model = trainNewModel(regKwargs, classKwargs)
    # model = Model("XGB_regressor.xgb", "new classifier.xgb")
    return evaluate(model)
def evaluate(model: Model):
    x, y = evalData()
    numChecks = 100
    for i in range(5, numChecks - 5, 5):
        threshold = float(i) / numChecks
        prediction = model.predict(x, threshold=threshold)
        metrics = evaluateMetrics(prediction, y)
        print(f'Threshold: {threshold}', metrics)
    return metrics

if __name__ == '__main__':
    kwargsInputs = {
        'class_colsample_bytree': 0.3, 'class_learning_rate': 0.1,
        'class_max_depth': 200, 'class_alpha': 10, 'class_n_estimators': 70, 'class_scale_pos_weight': 70
    }
    kwargs = defaultdict(lambda: None)
    for key in kwargsInputs:
        kwargs[key] = kwargsInputs[key]
    trainAndEvaluate(kwargs)



