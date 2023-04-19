import numpy as np
import pandas as pd
import json
import os
from os import walk
from openpyxl import load_workbook
from datetime import datetime



if __name__ == '__main__':
    path = 'C:\\Users\\JLanini\\OneDrive - DOI\\Documents\\GitHub\\Ruedi-RW\\Data\\'

    f_in=path+"Diversion SDI.xlsx"

    print('Processing ' + f_in)
    df = pd.read_excel(f_in,converters={'AltName': str})
    dfnew=pd.DataFrame()
    dfnew["ObjectSlot"]=df['Object']+'.'+df['Slot']
    dfnew['AltName']=df['AltName']
    dfnew.columns=["ObjectSlot","AltName"]
    df_json=dfnew.to_json(orient='records')
    # output
    with open(path+"HDB_SDI.json", 'w') as f:
        json.dump(df_json, f)

    # Closing file
    f.close()
