"""
@Author Ben Podrazhansky
"""

import sqlite3
import time
from datetime import datetime
from Data import NormalDBPath, AttackDBPath, ALL_COLUMNS, DATE_FORMAT, TIME_STEP
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from math import floor

class SequencedDataIterator(object):
    def __init__(self, batchSize, sequenceLength: int, dbPath: str, tableName: str, includeData: bool,
                 includeLabel: bool):
        super(SequencedDataIterator, self).__init__()
        if not includeData and not includeLabel:
            raise Exception("At least one of includeData or includeLabel must be True.")
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.dbPath = dbPath
        self.tableName = tableName
        self.includeData = includeData
        self.includeLabel = includeLabel

        con = sqlite3.connect(self.dbPath)
        cursor = con.cursor()
        firstTimeStamp = list(cursor.execute("SELECT {0} FROM {1} LIMIT 1".format(ALL_COLUMNS[0], self.tableName)))[0][
            0]
        self.numRows = list(cursor.execute("SELECT COUNT({0}) FROM {1}".format(ALL_COLUMNS[0], self.tableName)))[0][0]
        cursor.close()
        con.close()
        self.firstDate = datetime.strptime(firstTimeStamp, DATE_FORMAT) - TIME_STEP
        self.lastDate = datetime.strptime(firstTimeStamp, DATE_FORMAT) - TIME_STEP
        if self.batchSize > self.numRows:
            self._len = 1
        elif self.batchSize == float('inf'):
            self._len = 1
        else:
            self._len = floor(self.numRows / self.batchSize) + (self.batchSize % self.sequenceLength)
    def __len__(self):
        return self._len
    def __call__(self, *args, **kwargs):
        return self.__next__()



    def reset(self):
        """
        Resets the iterator to the beginning of the data
        """
        self.lastDate = self.firstDate

    def to_numpy(self):
        seqs = []
        if self.includeData:
            seqs.append(
                SequencedDataIterator(float('inf'), self.sequenceLength, self.dbPath, self.tableName, True,
                                      False).selectNextNRows(float('inf')).__next__())
        if self.includeLabel:
            seqs.append(
                SequencedDataIterator(1, self.sequenceLength, self.dbPath, self.tableName, False, True).selectNextNRows(
                    float('inf')).__next__().squeeze())

        if len(seqs) == 1:
            return seqs[0]
        else:
            return seqs[0], seqs[1]

    def __iter__(self):
        self.reset()
        return self

    def selectNextNRows(self, numRows):
        dataCols = ", ".join(ALL_COLUMNS[1:-1])
        labelCol = ALL_COLUMNS[-1]
        cols = []
        if self.includeData:
            cols.append(dataCols)
        if self.includeLabel:
            cols.append(labelCol)
        for colString in cols:
            queryString = "SELECT {0} FROM {1}".format(colString, self.tableName)
            if self.lastDate is not None:
                queryString += """
                        WHERE Timestamp > datetime('{0}')""".format(self.lastDate)
            queryString += """
                    ORDER BY Timestamp ASC"""
            if numRows < float('inf'):
                queryString += " LIMIT {0}".format(numRows)
            con = sqlite3.connect(self.dbPath)
            cursor = con.cursor()
            data = list(cursor.execute(queryString))
            cursor.close()
            con.close()
            if len(data) == 0:
                raise StopIteration
            yield np.array(data)
        self.lastDate += len(data) * TIME_STEP

    def __next__(self):
        numRows = self.sequenceLength + self.batchSize - 1
        resultRows = self.selectNextNRows(numRows)
        try:
            if self.includeData:
                if self.includeLabel:
                    data, labels = resultRows
                else:
                    data = resultRows
            elif self.includeLabel:
                labels = resultRows

            if self.includeData and self.includeLabel:
                datRet = sliding_window_view(data, (self.sequenceLength, data.shape[-1])).squeeze()
                labelRet = sliding_window_view(labels, (self.sequenceLength, labels.shape[-1])).squeeze().max(-1)
                return datRet, labelRet
            if self.includeData:
                # Data
                data = np.array(list(data)).squeeze(0)
                datRet = sliding_window_view(data, (self.sequenceLength, data.shape[-1])).squeeze()
                return datRet
            if self.includeLabel:
                # Labels
                labels = np.array(list(labels)).squeeze(0)
                labelRet = sliding_window_view(labels, (self.sequenceLength, labels.shape[-1])).squeeze().max(-1)
                return labelRet
        except:
            raise StopIteration

def getNormalData():
    """
    :return: An numpy ndarray of the normal data
    """
    return SequencedDataIterator(float('inf'), 1, NormalDBPath(), "Normal", True, False).to_numpy()
def getAttackData():
    """
    :return: two numpy ndarrays. First with the column data, and the second with the label data
    """
    return SequencedDataIterator(1, 1, AttackDBPath(), "Attack", True, True).to_numpy()

def getNormalDataIterator(batchSize, sequenceLength: int, includeData=False, includeLabel=False):
    """
    :param includeData: If False, will only iterate through the labels
    :param includeLabel: If true, will return tuple (train batch, label batch) when iterated through. Otherwise will exclude label
    :param batchSize:
    :param sequenceLength:
    :return: Returns an iterator throught the normal data
    """
    if not includeData and not includeLabel:
        raise Exception("At least one of includeData or includeLabel must be True.")
    return SequencedDataIterator(batchSize, sequenceLength, NormalDBPath(), "Normal", includeData, includeLabel)


def getAttackDataIterator(batchSize, sequenceLength: int, includeData=False, includeLabel=False):
    """
    :param includeData: If False, will only iterate through the labels
    :param includeLabel: If true, will return tuple (train batch, label batch) when iterated through. Otherwise will exclude label
    :param batchSize:
    :param sequenceLength:
    :return: Returns and iterator through the attack data
    """
    if not includeData and not includeLabel:
        raise Exception("At least one of includeData or includeLabel must be True.")
    return SequencedDataIterator(batchSize, sequenceLength, AttackDBPath(), "Attack", includeData, includeLabel)


if __name__ == '__main__':
    iterator = getAttackDataIterator(1000, 100)
    index = 0
    times = []
    curr = time.time()
    for train, label in iterator:
        newCurr = time.time()
        diff = newCurr - curr
        times.append(diff)
        print("Batch {0}\tTime: {1}\t Avg: {2}".format(index, round(diff, 5), round(np.average(times), 5)))
        index += 1
        curr = time.time()
