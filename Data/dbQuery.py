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
    def __init__(self, batchSize: int, sequenceLength: int, dbPath: str, tableName: str):
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.dbPath = dbPath
        self.tableName = tableName

        con = sqlite3.connect(self.dbPath)
        cursor = con.cursor()
        firstTimeStamp = list(cursor.execute("SELECT {0} FROM {1} LIMIT 1".format(ALL_COLUMNS[0], self.tableName)))[0][0]
        cursor.close()
        con.close()

        self.lastDate = datetime.strptime(firstTimeStamp, DATE_FORMAT) - TIME_STEP

    def __iter__(self):
        return self

    def selectNextNRows(self, numRows):
        dataCols = ", ".join(ALL_COLUMNS[1:-1])
        labelCol = ALL_COLUMNS[-1]
        for colString in [dataCols, labelCol]:
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
        data, labels = self.selectNextNRows(numRows)
        # Data
        yield sliding_window_view(data, (self.sequenceLength, data.shape[-1])).squeeze()
        # Labels
        yield sliding_window_view(labels, (self.sequenceLength, labels.shape[-1])).squeeze().max(-1)


def getNormalDataIterator(batchSize: int, sequenceLength: int):
    """

    :param batchSize:
    :param sequenceLength:
    :return: Returns an iterator throught the normal data
    """
    return SequencedDataIterator(batchSize, sequenceLength, NormalDBPath(), "Normal")


def getAttackDataIterator(batchSize: int, sequenceLength: int):
    """
    :param batchSize:
    :param sequenceLength:
    :return: Returns and iterator through the attack data
    """
    return SequencedDataIterator(batchSize, sequenceLength, AttackDBPath(), "Attack")


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
