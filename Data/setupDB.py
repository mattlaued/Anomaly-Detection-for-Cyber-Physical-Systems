"""
Author: @BPod123

This script reloads the database tables listed below

"""
import pandas as pd
import sqlite3

if __name__ == '__main__':
    chunkSize = 10000
    con = sqlite3.connect("Data.db")
    cursor = con.cursor()
    for name, skiprows in [["SWaT_Dataset_Normal_v0", [0]], ["SWaT_Dataset_Normal_v1" , [0]], ["SWaT_Dataset_Attack_v0", []]]:
        cursor.execute("DROP TABLE IF EXISTS {0}".format(name))
        linesWritten = 0
        reader = pd.read_csv(name + ".csv", skiprows=skiprows, chunksize=chunkSize, parse_dates=[0])
        for chunk in reader:
            chunk.to_sql(name, con, if_exists='append', index=True)
            linesWritten += chunkSize
            print("Lines Written to {0}: {1}".format(name, linesWritten))

        # Remove space from and capitalize ' Timestamp' and change 'Normal/Attack' to 'NORMAL_ATTACK'
        # Slashes and spaces are bad practice in sql column names
        cursor.execute(
            """ALTER TABLE {0}
            RENAME COLUMN ' Timestamp' TO 'TIMESTAMP'""".format(name))
        cursor.execute(
            """ALTER TABLE {0}
            RENAME COLUMN 'index' TO 'ID'""".format(name))
        cursor.execute(
            """ALTER TABLE {0}
            RENAME COLUMN 'Normal/Attack' TO 'NORMAL_ATTACK'""".format(name))
    cursor.close()
    # Save changes to the database
    con.commit()
    con.close()
