"""
@Author Ben Podrazhansky
"""

import sqlite3
from Data import columns, NormalDBPath, AttackDBPath

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



# def __getSequence(table, start, end):
#     """
#     Local function to help with getting segments of the data
#     :param table: name of table
#     :param start: start index for data (inclusive)
#     :param end: end index for data (exclusive)
#     :return:
#     """
#     con = sqlite3.connect(__pathToDB())
#     cursor = con.cursor()
#     # rows = list(cursor.execute("SELECT * FROM {0} WHERE ID >= {1} AND ID < {2}".format(table, start, end)))
#     rows = list(cursor.execute("SELECT * FROM {0} WHERE ID >= {1} AND ID < {2}".format(table, start, end)))
#     """
#     CREATE TABLE {0} (
#     ID INTEGER PRI
#     """
#
#     z = 3
# def getSequencedData(table:str, sequenceLength:int):
#     """
#
#     :param table: Name of the table that you want to
#     :param sequenceLength: How many time stamps you want in each segment of data
#     :return: Two arrays. The first of the shape (len(Data) - sequenceLength, sequenceLength, (number of columns (excluding labels)))
#      containing the training
#      The second is of the form (len(Data) - sequenceLength, sequenceLength) containing the labels
#     """
#     con = sqlite3.connect(__pathToDB())
#     cursor = con.cursor()
#     numRows = list(cursor.execute("SELECT COUNT(*) FROM {0}".format(table)))[0][0]
#     cursor.close()
#     con.close()
#     # numRows = list(cursor.execute("SELECT COUNT(*) FROM {0}".format(table)))[0][0]
#     indexGen = ((i, i + sequenceLength) for i in range(0, numRows, sequenceLength))
#     for start, end in indexGen:
#         x = __getSequence(table, start, end)
#         z = 3
#     z = 3
#     pass
if __name__ == '__main__':
    iterator = getNormalDataIterator(5, 10)
    stop = 5
    index = 0
    for train, label in iterator:
        print("Batch {0} Train\n{1}\nBatch {0}\n\nLabel\n{2}".format(index, train, label))
        index += 1
        if index == stop:
            break



