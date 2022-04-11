"""
@Author Ben Podrazhansky
"""

import sqlite3
import time
from datetime import datetime
from Data import NormalDBPath, AttackDBPath, ALL_COLUMNS, DATE_FORMAT, TIME_STEP
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np


class SequencedDataIterator(object):
    def __init__(self, batchSize, sequenceLength: int, dbPath: str, tableName: str, includeData: bool,
                 includeLabel: bool):
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
        cursor.close()
        con.close()
        self.firstDate = datetime.strptime(firstTimeStamp, DATE_FORMAT) - TIME_STEP
        self.lastDate = datetime.strptime(firstTimeStamp, DATE_FORMAT) - TIME_STEP

    def reset(self):
        """
        Resets the iterator to the beginning of the data
        """
        self.lastDate = self.firstDate

    def getAllRemaining(self):
        seq = SequencedDataIterator(float('inf'), self.sequenceLength, self.dbPath, self.tableName, self.includeData,
                                    self.includeLabel)
        seq.lastDate = self.lastDate
        allRemaining = seq.__next__()
        return allRemaining

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
        if self.includeData:
            if self.includeLabel:
                data, labels = resultRows
            else:
                data = resultRows
        elif self.includeLabel:
            labels = resultRows
        else:
            raise Exception(
                "Both self.includeData and self.includeLabel are False. You messed up creating the iterator")
        try:
            if self.includeData and self.includeLabel:
                return sliding_window_view(data, (self.sequenceLength, data.shape[-1])).squeeze(), sliding_window_view(
                    labels, (self.sequenceLength, labels.shape[-1])).squeeze().max(-1)
            if self.includeData:
                # Data
                data = np.array(list(data)).squeeze(0)
                return sliding_window_view(data, (self.sequenceLength, data.shape[-1])).squeeze()
            if self.includeLabel:
                # Labels
                labels = np.array(list(labels)).squeeze(0)
                return sliding_window_view(labels, (self.sequenceLength, labels.shape[-1])).squeeze().max(-1)
        except:
            raise StopIteration


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
