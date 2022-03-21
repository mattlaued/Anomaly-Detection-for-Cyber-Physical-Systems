"""
@Author Ben Podrazhansky
"""

import sqlite3
from Data import NormalDBPath, AttackDBPath

class SequencedDataIterator(object):
    def __init__(self, batchSize:int, sequenceLength:int, dbPath:str, tableName:str):
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.dbPath = dbPath
        self.tableName = tableName
        self.lastDate = None
    def __iter__(self):
        return self
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


def getNormalDataIterator(batchSize:int, sequenceLength:int):
    """

    :param batchSize:
    :param sequenceLength:
    :return: Returns an iterator throught the normal data
    """
    return SequencedDataIterator(batchSize, sequenceLength, NormalDBPath(), "Normal")
def getAttackDataIterator(batchSize:int, sequenceLength:int):
    """
    :param batchSize:
    :param sequenceLength:
    :return: Returns and iterator through the attack data
    """
    return SequencedDataIterator(batchSize, sequenceLength, AttackDBPath(), "Attack")

if __name__ == '__main__':
    iterator = getNormalDataIterator(5, 10)
    stop = 5
    index = 0
    for train, label in iterator:
        print("Batch {0} Train\n{1}\nBatch {0}\n\nLabel\n{2}".format(index, train, label))
        index += 1
        if index == stop:
            break



