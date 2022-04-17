from outliers import smirnov_grubbs as grubbs
import numpy as np

try:
    # If you don't have a cuda capable card and the dependencies installed separatly, you will not be able to use
    # cupy
    import cupy as cp
except:
    pass
from Data import ALL_COLUMNS
from Data.setupDB import AttackDBPath, NormalDBPath
from concurrent.futures import ProcessPoolExecutor
import sqlite3




def createConnection(dbPathTables: dict, dbUsePath, mainTableName):
    tableNames = []
    con = sqlite3.connect(dbUsePath)
    cursor = con.cursor()
    dbIndex = 0
    for dbPath in dbPathTables:
        dbIndex += 1
        tableIndex = 0
        for tableName in dbPathTables[dbPath]:
            tableIndex += 1
            cursor.execute(f"ATTACH DATABASE '{dbPath}' AS db_{dbIndex}")
            tableNames.append(f"'db_{dbIndex}'.{tableName}")
    del dbPath, dbIndex, tableName, tableIndex
    cursor.close()
    return con, tableNames, mainTableName


def TietjenStatistic_DB(cols, kRange, dbPathTables: dict, dbUsePath, mainTableName, module=np):
    """
    :param cols: columns of database to calculate the Tietjen Statistic for
    :param kRange: Range of k-values to calculate the statistic for
    :param dbPathTables: Dict: {pathToDb: [relevant table names in database]}
    The tables are unioned for this operation.
    :param dbUsePath: The path of the database to use for this
    :return: array of shape (|cols|, |kRange|) with the statistic value for each column
    """
    con, tableNames, mainTableName = createConnection(dbPathTables, dbUsePath, mainTableName)
    cursor = con.cursor()
    fromString = "(" + "\nUNION\n".join([f"""
    SELECT {", ".join(cols)} FROM {tableName}
    """ for tableName in tableNames]) + ")"
    avgsAndCount = list(cursor.execute(
        """
        SELECT {0}, COUNT({1}) FROM {2};
        """.format(", ".join([f"AVG({col})" for col in cols]), cols[0], fromString)))[0]
    count = avgsAndCount[-1]
    avgs = avgsAndCount[:-1]
    del avgsAndCount


    z_col_str = lambda col, avg: f"""
            SELECT {col}_z
            FROM (
            SELECT {col} AS {col}_z, ABS({col} - {avg}) AS {col}_r
            FROM {fromString}
                ORDER BY {col}_r ASC
               )
    """
    cursor.execute("""DROP TABLE IF EXISTS Tietjen_Statistics"""),
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS Tietjen_Statistics(
            COL TEXT,
            K INTEGER,
            VALUE REAL,
            PRIMARY KEY (COL, K));""")

    cursor.close()
    con.commit()
    con.close()
    kList = list(kRange)
    kList.sort()
    kList = list(reversed(kList))

    queryString = lambda col, avg: f"""
                        SELECT {col}_z
                        FROM ({z_col_str(col, avg)})
                        """
    maxWorkers = 5
    executor = ProcessPoolExecutor(max_workers=maxWorkers)

    conInfo = [dbPathTables, dbUsePath, mainTableName]
    # This WILL cause problems on computers with less ram if max_workers is too high.
    futures = []
    for col, avg in zip(cols, avgs):
         futures.append(executor.submit(TienjenStatistic, conInfo, col, queryString(col, avg), kList, module))
        # future.add_done_callback(futureCallback)
    executor.shutdown()
    for future in futures:
        args = future.result()
        InsertIntoTable(*args)



def InsertIntoTable(npE, kRange, col, conInfo):
    npE, kRange, col, conInfo
    con, tableNames, mainTableName = createConnection(conInfo[0], conInfo[1], conInfo[2])
    cursor = con.cursor()
    cursor.execute(f"""
        INSERT INTO {mainTableName} (COL, K, VALUE)
        VALUES {', '.join(map(lambda k_i: f"('{col}', {str(k_i[0])}, {str(npE[k_i[1]])})", zip(kRange, range(len(kRange)))))};
        """)
    cursor.close()
    con.commit()
    con.close()
    # breakpoint()


def TienjenStatistic(conInfo, col, queryString, kRange, module):
    module = np if module == 'np' else cp
    con, tableNames, mainTableName = createConnection(conInfo[0], conInfo[1], conInfo[2])
    cursor = con.cursor()
    cursor.execute(queryString)
    z = module.array(list(cursor)).squeeze()
    cursor.close()
    con.close()

    total = None
    denom = ((z - z.mean()) ** 2).sum()
    if denom != 0:
        E = []
        for k in kRange:
            if total is None:
                total = z[:-(k + 1)].sum()
            total += z[-(k + 1)]
            mean = total / (len(z) - k)
            E.append(module.sum((z[:-k] - mean) ** 2) / denom)
        del z, total, denom, k, mean
        if module != np:
            cpE = module.array(E)
            npE = module.asnumpy(cpE)
            del cpE
            # return npE, kRange, col, conFunc
        else:
            npE = np.array(E)
    else:
        npE = np.zeros(len(kRange))

    return npE, kRange, col, conInfo





if __name__ == '__main__':
    # Number of attack rows: 5484
    # Columns to test: AIT501, AIT503, FIT502
    dbTableDict = {NormalDBPath(): ["Normal"], AttackDBPath(): ["Attack"]}
    try:
        module = 'cp'
    except:
        module = 'np'
    tietJenStatistics = TietjenStatistic_DB(ALL_COLUMNS[1:-1], range(1, 20000), dbTableDict, "Tietjen.db",
                                            "Tietjen_Statistics", module='cp')

