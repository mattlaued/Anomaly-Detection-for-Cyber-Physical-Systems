"""
@Author Ben Podrazhansky
"""

import sqlite3
from Data import NormalDBPath, AttackDBPath


class SequencedDataIterator(object):
    def __init__(self, batchSize: int, sequenceLength: int, dbPath: str, tableName: str):
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.dbPath = dbPath
        self.tableName = tableName
        self.lastDate = None

    def __iter__(self):
        return self

    def nextNPoints(self, n: int, sequenceLength=None):
        """
        You can set n to float("inf") to receive all of the remaining data points
        :param n: Number of data points to get
        :param sequenceLength: Defaults to self.sequenceLength
        :return: Returns the next n batches
        """
        if sequenceLength is None:
            sequenceLength = self.sequenceLength
        newIt = SequencedDataIterator(n, sequenceLength, self.dbPath, self.tableName)
        newIt.lastDate = self.lastDate
        next = newIt.__next__()
        self.lastDate = newIt.lastDate
        return next

    def __next__(self):
        con = sqlite3.connect(self.dbPath)
        cursor = con.cursor()
        numRows = self.sequenceLength + self.batchSize - 1
        if self.lastDate is None:
            data = list(
                cursor.execute(
                    """
                    SELECT * FROM {0}
                    ORDER BY Timestamp ASC LIMIT {1}""".format(self.tableName, numRows)))
        else:
            data = list(cursor.execute(
                """
                SELECT * FROM {0}
                WHERE Timestamp > datetime('{2}')
                ORDER BY Timestamp ASC LIMIT {1}
                  """.format(self.tableName, numRows, self.lastDate)))
        if len(data) == 0:
            return [], []
        self.lastDate = data[-1][0]
        batchSize = self.batchSize - (self.batchSize + self.sequenceLength - 1 - len(data))
        batchTrain = tuple(
            [data[i + j][1:-1] for j in range(self.sequenceLength)]
            for i in range(batchSize)
        )
        batchLabel = tuple(
            [data[i + j][-1] for j in range(self.sequenceLength)]
            for i in range(batchSize)
        )
        return batchTrain, batchLabel


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
    iterator = getAttackDataIterator(5, 10)
    stop = 5
    index = 0
    tenBatches = iterator.nextNBatches(10)
    for train, label in iterator:
        print("Batch {0} Train\n{1}\nBatch {0} Label\n{2}\n".format(index, train, label))
        index += 1
        if index == stop:
            break
