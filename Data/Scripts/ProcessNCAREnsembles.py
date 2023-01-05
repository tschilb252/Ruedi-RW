from numpy.core.fromnumeric import trace
import pandas as pd
import json
import urllib.request
from pandas.core import series
from pandas.core.indexes.base import Index
import spotpy
from spotpy import analyser
from spotpy import objectivefunctions as of 
from pandas.tseries.offsets import DateOffset
import numpy as np
import csv
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt 
import os
from os import walk
import isodate
def getHydromet(sta,parm,sd,ed): #Station code, parameter code, start date, end date
    #function obtains hydromet data from webservice.  req
    URL = 'https://www.usbr.gov/gp-bin/webarccsv.pl?parameter='+sta+'%20'+parm+'&syer='+str(sd.year)+'&smnth='+str(sd.month)+'&sdy='+str(sd.day)+'&eyer='+str(ed.year)+'&emnth='+str(ed.month)+'&edy='+str(ed.day)+'&format=10'
    #df = pd.read_csv(URL,header=14,index_col=0,parse_dates=True)
    #df = pd.read_json(URL)
    f = urllib.request.urlopen(URL)
    #data=f.read()
    #print(data)
    data=json.load(f)
    d=data['SITE']['DATA']
    df = pd.json_normalize(d)
    df.set_index('DATE', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df
def write_trace(df,tracenum,Head_Dict): #function to write the trace directory files
    directory=outdir+df.index[0].strftime("%Y%m%d")+'\\trace'+str(tracenum)
    fname=directory+'\\Buffalo_Bill_Inflows.local.rdf'
    #write the trace directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fname, 'w') as fout:
        # writer = csv.writer(fout)
        # for k, v in Head_Dict.items():
        #     writer.writerow(v)
        for key in Head_Dict.keys():
            fout.write("%s\n" % (Head_Dict[key]))
        if(isinstance(tracenum, int)):
            for index, row in df.iterrows():
                fout.write("%f\n" % (row[tracenum-1]))
        else:
            for index, row in df.iterrows():
                fout.write("%f\n" % (row[tracenum]))
        #df[column].to_csv(fname,index=None,header=False,mode='a+')
#Create a header dictionary:
Head_Dict={'sd':'start_date: 2017-10-01 24:00',
            'ed':'end_date: 2021-06-30 24:00',
            'ts':'timestep: 1 DAY',
            'un':'units: cfs',
            'sc':'scale: 1.000000',
            'slot':'# Series Slot: Buffalo Bill Inflows.Local Inflow [1 cfs]'}
#conversion factor for cms to cfs
cms2cfs=35.3146667
#Output Directory top level
outdir="C:\\Projects\\BBR\\NCAR_ENS\\TraceDir\\"
#Read the directory structure
#Set top-level path with xml enemble files to loop thru
mypath=("C:\\Projects\\BBR\\NCAR_ENS\\Data")
os.chdir(mypath)
obs=getHydromet('BBR','IN',pd.to_datetime('1990-01-01'),pd.to_datetime('2021-05-01')) #get the observations for the period
#open a csv containing historical inflows-necessary to write data to beginning of file for input
#hist_df = pd.read_csv('hist.csv', parse_dates=True, index_col=0)
f = []
#make figure to plot forecasts
plt.plot(obs, linewidth=3)
kge=pd.DataFrame(columns=['mean', 'median'])
for root, dirs, files in walk(mypath,topdown=True):
    for x in files:
        if (x.endswith(".blend.csv") and not x.endswith(".BC.blend.csv")):
            #print('Processing '+x)
            #read file into dataframe
            df = pd.read_csv(root+'\\'+x, parse_dates=True, index_col=0)
            #convert from cms to cfs
            df=df*cms2cfs
            #get start date and end date and write to header dictionary
            Head_Dict['sd']='start_date: '+df.index[0].strftime("%Y-%m-%d")+' 24:00'
            Head_Dict['ed']='end_date: '+df.index[-1].strftime("%Y-%m-%d")+' 24:00'
            #write each trace to a file
            for column in df:
                #print(Head_Dict['sd']," ", column,df[column].min())
                if (df[column].min()<1):
                    print(Head_Dict['sd']," ", column,df[column].min())
                    #If we have zeros, overwrite with previous trace
                    df[column]=df.iloc[:, [df.columns.get_loc(column)-1]]
                write_trace(df,df.columns.get_loc(column)+1,Head_Dict)
            #Create Mean and Median Traces
            df_copy=df.copy()
            df['mean']=df_copy.mean(numeric_only=True, axis=1)
            df['median']=df_copy.median(numeric_only=True, axis=1)
            write_trace(df,'mean',Head_Dict)
            write_trace(df,'median',Head_Dict)
            #window the observation
            mask = (obs.index >= df.index[0]) & (obs.index <= df.index[-1])
            eval=obs[mask].iloc[:, 0]
            #kge.loc[df.index[0]] = of.kge(eval.values.tolist(),df['mean'].values.tolist()),of.kge(eval.values.tolist(),df['median'].values.tolist())
            kge.loc[df.index[0]] = of.kge_non_parametric(eval.values.tolist(),df['mean'].values.tolist()),of.kge_non_parametric(eval.values.tolist(),df['median'].values.tolist())
            #add mean to plot
            #plt.plot(df['mean'],linewidth=.5)
pltstart=pd.to_datetime('2017-10-01')
pltend=pd.to_datetime('2019-09-30')
plt.xlim(pltstart,pltend)
#plt.show()
#calculate rolling average

kge=kge.sort_index(ascending=True)
#resample to daily 
kge2=kge.resample('D').mean()
#kge=kge.interpolate()
#ave=kge.rolling(7).mean()
#calculate the average by day of year
#mask = (kge.index.dt.is_leap_year == 1) & (kge.idx.dt.dayofyear == 60)
ann_ave=kge2.groupby([kge2.index.month, kge2.index.day]).mean()
ann_ave=ann_ave.reset_index(level=[0,1]).dropna()

""" #plot total timeseries and daily average
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Kling-Gupta Efficiency (Non-Parametric')
ax1.plot(kge.index, kge['mean'])
ax1.plot(kge.index, kge['median'])
ax1.set_ylabel('KGE')
ax2.plot(ann_ave.index,ann_ave['mean'])
ax2.plot(ann_ave.index,ann_ave['median'])
ax2.set_xlabel('Day of Year')
ax2.set_ylabel('KGE')
ax1.legend(['Ensemble Mean, KGE= ' + str(kge['mean'].mean())[:4],'Ensemble Median, KGE= ' + str(kge['median'].mean())[:4]],
    loc='upper center',bbox_to_anchor=(0.5, 1.2), shadow=True, ncol=2)
#plt.legend(['Ensemble Mean, KGE= ' + str(kge['mean'].mean())[:4],'Ensemble Median, KGE= ' + str(kge['median'].mean())[:4]])
plt.show()
fig.savefig('test.jpg')
plt.hist(kge['median']-kge['mean'])
plt.legend(['Median minus Mean KGE'])
plt.show()
# Load the covariate data
        #covariate_data = pd.read_csv('covariate_data.csv', parse_dates=True, index_col=0)
 """
