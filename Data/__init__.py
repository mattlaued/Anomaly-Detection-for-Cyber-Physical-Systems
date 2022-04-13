ALL_COLUMNS = ['Timestamp', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603', 'ATTACK']
REAL_COLUMNS = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'DPIT301', 'FIT301', 'LIT301', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'PIT501', 'PIT502', 'PIT503', 'FIT601']
" 28/12/2015 10:00:00 AM"
# DATE_FORMAT = " %d/%m/%Y %I:%M:%S %p" # See datetime.datetime.strptime and datetime.datetime.strftime format codes
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# import datetime
from datetime import timedelta
TIME_STEP = timedelta(seconds=1)
# def timestampToDateTime(string: str):
#     datetime.datetime.strptime(string, DATE_FORMAT)


from Data.setupDB import NormalDBPath, AttackDBPath
from Data.dbQuery import getAttackDataIterator, getNormalDataIterator, SequencedDataIterator, getNormalData, getAttackData
