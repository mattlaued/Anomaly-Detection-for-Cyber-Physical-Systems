import pandas as pd
import sqlite3
if __name__ == '__main__':
    chunkSize = 10000
    con = sqlite3.connect("Data.db")
    for name in "SWaT_Dataset_Normal_v0 SWaT_Dataset_Normal_v1 SWaT_Dataset_Attack_v0".split(" "):
        linesWritten = 0
        reader = pd.read_csv(name + ".csv", skiprows=[0], chunksize=chunkSize, parse_dates=[0], index_col=False)
        for chunk in reader:
            chunk.to_sql(name, con, if_exists='append')
            linesWritten += chunkSize
            print("Lines Written to {0}: {1}".format(name, linesWritten))
    con.close()