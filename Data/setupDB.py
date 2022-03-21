"""
Author: Ben Podrazhansky

This script reloads the database by creating a separate database for each table in parallel then attaches them to a
single database. Next the data is preprocessed. At the time or writing, the data is preprocessed by standardization.

"""
import pandas as pd
import sqlite3
import time
from threading import Thread, Lock
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import sys
import math
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
        return math.sqrt(self.S / (self.k-2))

def __pathToDB():
    """
    Local function, used to get the path that is subject to change based on where the main method is started from
    :return: string path to Data.db
    """
    repoName = 'Anomaly-Detection-for-Cyber-Physical-Systems'
    currDir = os.getcwd()
    folders = currDir.split("\\") if "\\" in currDir else currDir.split("/")
    if repoName not in folders:
        raise "Path Error: Is the repo name not \'{0}\'?".format(repoName)
    folders = folders[:folders.index(repoName) + 1]
    dbPath = "/".join(folders) + "/Data/Data.db"
    return dbPath
def insertChunk(chunk:pd.DataFrame, tableName, linesWritten, dbPath, writeLock:Lock):
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
    # total = time.process_time() - start
    print("Lines Written to {0}:\t {1}\t\t+{2}".format(tableName, linesWritten[0], len(chunk)))


    # return "Lines Written to {0}: {1}".format(tableName, linesWritten[0])
def createTable(name, tableName, skipRows, tempDBPath):
    chunkSize = 5000
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

# def attachDatabases(tempDBPath, endDBPath, tableName, writeLock, asyncRes):
#     asyncRes.get()
#     writeLock.acquire()
#     con = sqlite3.connect(endDBPath)
#     cursor = con.cursor()
#     cursor.execute("""ATTACH DATABASE '{0}' AS {1};""".format(tempDBPath, tempDBPath[:tempDBPath.find(".")]))
#     cursor.close()
#     con.commit()
#     con.close()
#     writeLock.release()
#     print("Database {0} Attached to {1}".format(tempDBPath, endDBPath))
#     return time.time()
#






def createTables(start, endDBPath="Data.db"):
    fNames = ["SWaT_Dataset_Normal_v0", "SWaT_Dataset_Normal_v1", "SWaT_Dataset_Attack_v0"]
    tableNames = ["Normal_0", "Normal_1", "Attack"]
    skipRows = [[0], [0], []]
    # for name, tableName, skiprows in zip(fNames, tableNames, skipRows):
    #     if tableName != "Attack":
    #         continue
    #     reader = pd.read_csv(name + ".csv", skiprows=skiprows, chunksize=100, parse_dates=[0])
    #     for chunk in reader:
    #         insertChunk(chunk, tableName, [0], name + ".db", Lock())


    pool = Pool(int(min(len(fNames), cpu_count())))
    asyncResults = []
    for name, tableName, skiprows in zip(fNames, tableNames, skipRows):
        tempDBPath = "{0}.db".format(tableName)
        asyncResults.append(pool.apply_async(createTable, args=(name, tableName, skiprows, tempDBPath)))
    pool.close()
    pool.join()
    # writeLock = Lock()
    [asyncRes.get() for asyncRes in asyncResults]
    # con = sqlite3.connect(endDBPath)
    # cursor = con.cursor()
    # cursor.execute("ATTACH DATABASE 'Normal_0.db' as Normal_0")
    # cursor.execute("ATTACH DATABASE 'Normal_1.db' as Normal_1")
    # cursor.execute("ATTACH DATABASE 'Attack.db' as Attack")
    # con.commit()
    # cursor.close()
    # con.close()

    # Thread(target=asyncRes.get).start()
    # executor = ThreadPoolExecutor()
    # futures = [
    #     executor.submit(attachDatabases, tempDBPath, endDBPath, tableName, writeLock, asyncRes)
    #     for tempDBPath, tableName, asyncRes in asyncResults
    # ]

    # end = max([f.result() for f in futures])
    end = time.time() - start
    print("Time taken: {0}".format(end - start))
    sys.stdout.flush()
    return end


def combineAndStandardizeTables(start, lastEnd, dbPath):
    """
    The Normal Data was split up between two files that were mostly overlapping.
    This function combines the two sets of normal data into a single table, then removes the old tables
    """
    if lastEnd is None:
        lastEnd = time.time()
    # Create a table for the compliment of the intersection, then combine with one of the normal tables into a new table
    con = sqlite3.connect(dbPath)
    cursor = con.cursor()
    cursor.execute("ATTACH DATABASE 'Normal_0.db' as Normal_0")
    cursor.execute("ATTACH DATABASE 'Normal_1.db' as Normal_1")
    cursor.execute("ATTACH DATABASE 'Attack.db' as Attack")
    cursor.execute("DROP TABLE IF EXISTS NORMAL")
    cursor.execute("DROP TABLE IF EXISTS ATTACK")
    cursor.execute(
        """ CREATE TABLE NORMAL AS
        SELECT * FROM 'Normal_0'.Normal_0
        UNION
        SELECT * FROM 'Normal_1'.Normal_1
    """)
    # cursor.execute(
    #     """ CREATE TABLE ATTACK AS
    #     SELECT * FROM 'Attack'.Attack
    # """)
    # cursor.execute(
    #     """SELECT * INTO NORMAL FROM NORMAL_OUTLIERS OUTER JOIN Normal_0_db.Normal_0;
    #     """)

    standardizeCols = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'DPIT301', 'FIT301', 'LIT301', 'AIT402', 'FIT401',
     'LIT401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'PIT501', 'PIT502',
     'PIT503', 'FIT601']
    setString = ", ".join(["{0} = ({0} - AVG({0}))/stdev({0})".format(x) for x in standardizeCols])
    # con.create_aggregate("stdev", 1, StdevFunc)
    # for table in ["NORMAL"]:#, "ATTACK"]:
    #     cursor.execute("""
    #     UPDATE {0}
    #     SET {1}""".format(table, setString))

    end = time.time()

    z = 3


def preprocessTables():
    # con = sqlite3.connect("Data.db")
    # cursor = con.cursor()

    z = 3

def main(dbPath=None):
    start = time.time()
    end = None
    if dbPath is None:
        dbPath = __pathToDB()

    # end = createTables(start, dbPath)
    combineAndStandardizeTables(start, end, dbPath)
    preprocessTables()

if __name__ == '__main__':
    main()


