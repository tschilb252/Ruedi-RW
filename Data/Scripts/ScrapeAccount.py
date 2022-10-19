import numpy as np
import pandas as pd
import os
from os import walk
from openpyxl import load_workbook
from datetime import datetime


def get_file_names(mydir, str=''):
    # gets a list of the xml filenames in the directory for a particular station
    f = []
    f2 = []
    for root, dirs, files in walk(mydir, topdown=True):
        for x in files:
            if x.endswith(str + '.xlsx') or x.endswith(str + '.xls') or x.endswith(str + '.xlsm'):
                f.append(os.path.join(root, x))
    return f


if __name__ == '__main__':
    inpath = 'C:\\Users\\JLanini\\Documents\\GitHub\\Ruedi-RW\\Data\\Accounting Spreadsheets\\'
    outpath = 'C:\\Users\\JLanini\\Documents\\GitHub\\Ruedi-RW\\Data\\'
    sheetname = 'Contract Usage'
    skip = list(range(5, 46))
    dfout = pd.DataFrame()
    for f in get_file_names(inpath):
        print('Processing ' + f)
        df = pd.read_excel(f, sheet_name=sheetname, index_col=0, header=4, skiprows=skip, parse_dates=True)
        df.drop('Total Releases for Contractors', axis=1, inplace=True)
        dfout=dfout.append(df)
    dfout=dfout.sort_index()
    dfout.to_csv(outpath + 'AccountingDataTSAll.csv')
