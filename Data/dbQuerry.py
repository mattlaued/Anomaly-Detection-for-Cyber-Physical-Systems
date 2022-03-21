"""
@Author Ben Podrazhansky
"""

import sqlite3

from Data.setupDB import __pathToDB


def __getSequence(table, start, end):
    """
    Local function to help with getting segments of the data
    :param table: name of table
    :param start: start index for data (inclusive)
    :param end: end index for data (exclusive)
    :return:
    """
    con = sqlite3.connect(__pathToDB())
    cursor = con.cursor()
    # rows = list(cursor.execute("SELECT * FROM {0} WHERE ID >= {1} AND ID < {2}".format(table, start, end)))
    rows = list(cursor.execute("SELECT * FROM {0} WHERE ID >= {1} AND ID < {2}".format(table, start, end)))
    """
    CREATE TABLE {0} (
    ID INTEGER PRI
    """

    z = 3
def getSequencedData(table:str, sequenceLength:int):
    """

    :param table: Name of the table that you want to
    :param sequenceLength: How many time stamps you want in each segment of data
    :return: Two arrays. The first of the shape (len(Data) - sequenceLength, sequenceLength, (number of columns (excluding labels)))
     containing the training
     The second is of the form (len(Data) - sequenceLength, sequenceLength) containing the labels
    """
    con = sqlite3.connect(__pathToDB())
    cursor = con.cursor()
    numRows = list(cursor.execute("SELECT COUNT(*) FROM {0}".format(table)))[0][0]
    cursor.close()
    con.close()
    # numRows = list(cursor.execute("SELECT COUNT(*) FROM {0}".format(table)))[0][0]
    indexGen = ((i, i + sequenceLength) for i in range(0, numRows, sequenceLength))
    for start, end in indexGen:
        x = __getSequence(table, start, end)
        z = 3
    z = 3
    pass
if __name__ == '__main__':
    getSequencedData("SWat_Dataset_Normal_v0", 10)


