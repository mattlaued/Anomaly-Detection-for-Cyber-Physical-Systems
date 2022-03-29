"""
Author: Ben Podrazhansky

This script reloads the database by creating a separate database for each table in parallel then attaches them to a
single database. Next the data is preprocessed. At the time or writing, the data is preprocessed by standardization.

"""
from Data import REAL_COLUMNS
import pandas as pd
import sqlite3
import time
from threading import Thread, Lock
from multiprocessing import Pool, cpu_count, Process
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import math


def __pathToDB():
    """
    Local function, used to get the path that is subject to change based on where the main method is started from
    :return: string path to Data folder
    """
    repoName = 'Anomaly-Detection-for-Cyber-Physical-Systems'
    currDir = os.getcwd()
    folders = currDir.split("\\") if "\\" in currDir else currDir.split("/")
    if repoName not in folders:
        raise "Path Error: Is the repo name not \'{0}\'?".format(repoName)
    folders = folders[:folders.index(repoName) + 1]
    dbPath = "/".join(folders) + "/Data/"
    return dbPath


def NormalDBPath():
    return __pathToDB() + "Normal.db"


def AttackDBPath():
    return __pathToDB() + "Attack.db"


class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k - 2))


def standardize(x, mean, std):
    return (x - mean) / std


def insertChunk(chunk: pd.DataFrame, tableName, linesWritten, dbPath, writeLock: Lock):
    # start = time.process_time()
    con = sqlite3.connect(dbPath)
    chunk = chunk.rename(columns={col: col.replace(" ", "") for col in chunk.columns if col.replace(" ", "") != col})
    # [0 if chunk["Normal/Attack"][i] == "Normal" else 1 for i in range(len(chunk["Normal/Attack"]))]
    normalAttack = chunk["Normal/Attack"]
    attack = np.zeros_like(normalAttack)
    attack[normalAttack == "Attack"] = 1
    chunk.drop(columns=["Normal/Attack"], inplace=True)
    chunk.insert(len(chunk.keys()), "ATTACK", attack)

    writeLock.acquire()
    chunk.to_sql(tableName, con, if_exists='append', index=False)
    con.commit()
    writeLock.release()
    con.close()
    linesWritten[0] += len(chunk)
    print("Lines Written to {0}:\t {1}\t\t+{2}".format(tableName, linesWritten[0], len(chunk)))


def createTable(name, tableName, skipRows, tempDBPath):
    chunkSize = 10000
    con = sqlite3.connect(tempDBPath)
    cursor = con.cursor()
    cursor.execute("DROP TABLE IF EXISTS {0}".format(tableName))
    cursor.execute("CREATE TABLE {0}".format(tableName) + """(
    Timestamp TIMESTAMP,
    FIT101 REAL,
    LIT101 REAL,
    MV101 INTEGER,
    P101 INTEGER,
    P102 INTEGER,
    AIT201 REAL,
    AIT202 REAL,
    AIT203 REAL,
    FIT201 REAL,
    MV201 INTEGER,
    P201 INTEGER,
    P202 INTEGER,
    P203 INTEGER,
    P204 INTEGER,
    P205 INTEGER,
    P206 INTEGER,
    DPIT301 REAL,
    FIT301 REAL,
    LIT301 REAL,
    MV301 INTEGER,
    MV302 INTEGER,
    MV303 INTEGER,
    MV304 INTEGER,
    P301 INTEGER,
    P302 INTEGER,
    AIT401 INTEGER,
    AIT402 REAL,
    FIT401 REAL,
    LIT401 REAL,
    P401 INTEGER,
    P402 INTEGER,
    P403 INTEGER,
    P404 INTEGER,
    UV401 INTEGER,
    AIT501 REAL,
    AIT502 REAL,
    AIT503 REAL,
    AIT504 REAL,
    FIT501 REAL,
    FIT502 REAL,
    FIT503 REAL,
    FIT504 REAL,
    P501 INTEGER,
    P502 INTEGER,
    PIT501 REAL,
    PIT502 REAL,
    PIT503 REAL,
    FIT601 REAL,
    P601 INTEGER,
    P602 INTEGER,
    P603 INTEGER,
    ATTACK INTEGER,
    PRIMARY KEY (Timestamp)
            );
            """)
    cursor.execute("CREATE INDEX index_timestamp_{0} ON {0} (Timestamp);".format(tableName))
    con.commit()
    cursor.close()
    con.close()
    linesWritten = [0]

    reader = pd.read_csv(name + ".csv", skiprows=skipRows, chunksize=chunkSize, parse_dates=[0])

    tempWriteLock = Lock()
    [Thread(target=insertChunk, args=(chunk, tableName, linesWritten, tempDBPath, tempWriteLock)).start() for chunk in
     reader]


def createTables():
    fNames = ["SWaT_Dataset_Normal_v0", "SWaT_Dataset_Normal_v1", "SWaT_Dataset_Attack_v0"]
    tableNames = ["Normal_0", "Normal_1", "Attack"]
    skipRows = [[0], [0], []]

    pool = Pool(int(min(len(fNames), cpu_count())))
    asyncResults = []
    for name, tableName, skiprows in zip(fNames, tableNames, skipRows):
        tempDBPath = "{0}.db".format(tableName)
        asyncResults.append(pool.apply_async(createTable, args=(name, tableName, skiprows, tempDBPath)))
    pool.close()
    pool.join()
    [asyncRes.get() for asyncRes in asyncResults]


def combineNormalData():
    con = sqlite3.connect("Normal.db")
    cursor = con.cursor()
    cursor.execute("ATTACH DATABASE 'Normal_0.db' as Normal_0")
    cursor.execute("ATTACH DATABASE 'Normal_1.db' as Normal_1")
    cursor.execute("DROP TABLE IF EXISTS Normal")
    # Note that the UNION operation creates a set of DISTINCT elements. Therefore no duplicates will be included.
    cursor.execute(
        """ CREATE TABLE Normal AS
        SELECT * FROM 'Normal_0'.Normal_0
        UNION
        SELECT * FROM 'Normal_1'.Normal_1
    """)
    con.commit()
    cursor.close()
    con.close()
    # Remove Normal_0.db and Normal_1.db
    if os.path.isfile("Normal_0.db"):
        os.remove("Normal_0.db")
    if os.path.isfile("Normal_1.db"):
        os.remove("Normal_1.db")


# def columnAvg(dbPath, tableName, col):
#     con = sqlite3.connect(dbPath)
#     cursor = con.cursor()
#     val = list(cursor.execute("SELECT AVG({0}) FROM {1}".format(col, tableName)))[0][0]
#     cursor.close()
#     con.close()
#     return val


# def columnStDev(dbPath, tableName, col):
#     con = sqlite3.connect(dbPath)
#     cursor = con.cursor()
#     con.create_aggregate("stdev", 1, StdevFunc)
#     val = list(cursor.execute("SELECT stdev({0}) FROM {1}".format(col, tableName)))[0][0]
#     cursor.close()
#     con.close()
#     return val
def columnMeanStd(dbPath, tableName, col):
    con = sqlite3.connect(dbPath)
    cursor = con.cursor()
    mean = list(cursor.execute("SELECT AVG({0}) FROM {1}".format(col, tableName)))[0][0]
    con.create_aggregate("stdev", 1, StdevFunc)
    std = list(cursor.execute("SELECT stdev({0}) FROM {1}".format(col, tableName)))[0][0]
    cursor.close()
    con.close()
    return mean, std


def getNormalColumnMeanStds():
    """
    :return: The means and standard deviations of all real-valued columns in Normal.db
    """
    pool = Pool(int(min(cpu_count(), len(REAL_COLUMNS))))
    meanStdsAsync = {col: pool.apply_async(columnMeanStd, args=("Normal.db", "Normal", col)) for col in REAL_COLUMNS}
    pool.close()
    pool.join()
    meanStds = {col: meanStdsAsync[col].get() for col in REAL_COLUMNS}
    return meanStds


def standardizeTables(colMeanStds: dict):
    pool = Pool(int(min(cpu_count(), 2 * len(colMeanStds))))
    dbPaths = "Normal.db", "Attack.db"
    tableNames = "Normal", "Attack"
    [
        pool.apply_async(standardizeTable, args=(dbPath, tableName, colMeanStds))
        for dbPath, tableName in zip(dbPaths, tableNames)
    ]
    pool.close()
    pool.join()


def standardizeTable(dbPath, tableName, colMeanSTD):
    """
    :param colMeanSTD: Defaults to None, in which case it makes the following:
    A dict mapping column names to a value for means and standard deviations to use
            for that column.
    :return: A dict mapping column names to their mean and standard deviation.
    """

    # setStrings = []
    setString = ",\n".join([
        "{0} = standardize({0}, {1}, {2})".format(col, colMeanSTD[col][0], colMeanSTD[col][1])
        for col in colMeanSTD
    ])
    updateString = """UPDATE {0}
        SET {1};""".format(tableName, setString)

    con = sqlite3.connect(dbPath)
    cursor = con.cursor()
    con.create_function("standardize", 3, standardize)
    cursor.execute(updateString)
    con.commit()
    cursor.close()
    con.close()


if __name__ == '__main__':
    start = time.time()
    print("Creating Tables Normal_0, Normal_1, Attack")
    createTables()
    tableCreateTime = time.time()
    print("Created Tables Normal_0, Normal_1, Attack\nTime taken: {0}".format(round(tableCreateTime - start, 2)))

    print("Combining Tables Normal_0 and Normal_1, excluding duplicates and standardizing all data")
    print("Combining Normal_0 and Normal_1 data")
    combineNormalData()
    combineTime = time.time()
    print("Normal.db created. Normal_0.db and Normal_1.db removed")
    print("Time Taken: {0} (+{1})".format(round(combineTime - start, 2), round(combineTime - tableCreateTime, 2)))

    print("Calculating the mean and standard deviation of all real-valued columns in Normal data")
    colMeanStds = getNormalColumnMeanStds()
    meanStdTime = time.time()
    print("Normal real-valued column means and standard deviations calculated.")
    print("Time Taken: {0} (+{1})".format(round(meanStdTime - start, 2), round(meanStdTime - combineTime, 2)))

    print("Standardizing real-valued columns of Normal and Attack data with the Normal data means and standard "
          "deviations.")

    standardizeTables(colMeanStds)
    end = time.time()
    print("Normal and Attack data standardization complete")
    print("Time Taken: {0} (+{1})".format(round(end - start, 2),
                                          round(end - meanStdTime, 2)))
    #
    # print("Standardizing Normal Data")
    #
    # colMeanStd = standardizeTable("Normal.db", "Normal")
    # normalStandardizeTime = time.time()
    # print("Normal Data Standardized")
    # print("Real-valued column means and standard deviations recorded")
    # print("Time Taken: {0} (+{1})".format(round(normalStandardizeTime - start, 2),
    #                                       round(normalStandardizeTime - combineTime, 2)))
    # print("Standardizing Attack data")
    # # Use the mean and standard deviation of the normal data (6 days preceding the attack data)
    # # to standardize the attack data (5 days following the normal data).
    # standardizeTable("Attack.db", "Attack", colMeanStd)
    # end = attackStandardizeTime = time.time()
    # print("Attack Data Standardized")
    # print("Time Taken: {0} (+{1})".format(round(attackStandardizeTime - start, 2),
    #                                       round(attackStandardizeTime - normalStandardizeTime, 2)))
    #
    # print("Normal data stored in Normal.db.Normal")
    # print("Attack data stored in Attack.db.Attack")
    # print("Time: {0}".format(round(end - start, 2)))
