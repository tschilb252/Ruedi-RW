# Process ensemble hindcasts received from NWS
#pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <package-name> -U --user
import numpy as np
import pandas as pd
import properscoring as ps #package for CRPS
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.stats import norm
import xlsxwriter
#import graphviz
import matplotlib.pyplot as plt
import os
import tarfile
from os import walk
from xml.etree import ElementTree
import pint
from pint import UnitRegistry
cms2cfs=3.28084**3
#ureg = UnitRegistry()
#ureg.define('cfs=(ureg.feet**3)/ureg.second')
#ureg.define('cms=(ureg.meter**3)/ureg.second')
#ureg.define('acre=(ureg.feet*43560)')
cfsday2acft=24*60*60/43560
def rankz(obs,ensemble):
    ''' Parameters
    ----------
    obs : array of observations 
    ensemble : array of ensemble, with the first dimension being the 
        ensemble member and the remaining dimensions being identical to obs
    mask : boolean mask of shape of obs, with zero/false being where grid cells are masked.  
    Returns
    -------
    histogram data for ensemble.shape[0] + 1 bins. 
    The first dimension of this array is the height of 
    each histogram bar, the second dimension is the histogram bins. 
         '''


    
    combined=np.vstack((obs,ensemble))

    # print('computing ranks')
    ranks=np.apply_along_axis(lambda x: rankdata(x,method='min'),0,combined)

    # print('computing ties')
    ties=np.sum(ranks[0]==ranks[1:], axis=0)
    ranks=ranks[0]
    tie=np.unique(ties)

    for i in range(1,len(tie)):
        index=ranks[ties==tie[i]]
        # print('randomizing tied ranks for ' + str(len(index)) + ' instances where there is ' + str(tie[i]) + ' tie/s. ' + str(len(tie)-i-1) + ' more to go')
        ranks[ties==tie[i]]=[np.random.randint(index[j],index[j]+tie[i]+1,tie[i])[0] for j in range(len(index))]

    return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5, combined.shape[0]+1))

def computeMetrics(cv_yStar, yStar, yActual, p):
    # """ 
    # Computes statistical metrics using observed values vs forecasted values.
    # inputs:
    #     - yStar : forecastd / predicted values
    #     - yActual : actual / observed values
    #     - p : number of variables / predictors used to generate predicted values 

    # outputs:
    #     - {
    #         "Cross Validated Adjusted R2" : adjusted coefficient of determination using cross validated predictand
    #         "Root Mean Squared Prediction Error: root-mean-squared-error using cross validated predictand
    #         "Cross Validated Nash Sutcliffe" : Nash Sutcliffe model efficiency coefficient using cross validated predictand
    #         "Adjusted R2" : adjusted coefficient of determination using equation predictand,
    #         "Root Mean Squared Error" : root-mean-squared-error using equation predictand,
    #         "Nash-Sutcliffe" : Nash Sutcliffe model efficiency coefficient using equation predictand,
    #         "Sample Variance" : variance of the squared errors,
    #         "Mean Absolute Error" : mean of the absolute error values between observed and predicted values
    #     }

    # """

    yActual = np.array(yActual).flatten()
    yStar = np.array(yStar).flatten()
    cv_yStar = np.array(cv_yStar).flatten()

    """ Compute adjR2 values """
    yMean = float(np.mean(yActual))
    ssTotal = float(np.sum((yActual - yMean)**2))
    ssResidual = float(np.sum((yActual - yStar)**2))
    cv_ssResidual = float(np.sum((yActual - cv_yStar)**2))
    r2 = 1 - (ssResidual / ssTotal)
    cv_r2 = 1 - (cv_ssResidual / ssTotal)
    n = len(yActual)
    if n == (p+1):
        n = p + 1 + 0.00000001
    adjR2 = 1 - ((n-1)/(n-(p+1)))*(1-r2)
    cv_adjR2 = 1 - ((n-1)/(n-(p+1)))*(1-cv_r2)

    """ Compute the MAE """
    mae = float(np.sum(np.abs(yStar - yActual))) / (n-(p+1))

    """ Compute the RMSE """
    mse = ssResidual / (n-(p+1))
    cv_mse = cv_ssResidual / (n-(p+1))
    rmse = np.sqrt(mse)
    cv_rmse = np.sqrt(cv_mse)

    """ Compute the variance """
    s = ssTotal / (n-(p+1))

    """ Compute the nash-sutcliffe """
    numerator = np.sum((yStar - yActual)**2)
    cv_numerator = np.sum((cv_yStar - yActual)**2)
    denominator = np.sum((yActual - yMean)**2)
    nse = 1 - numerator / denominator
    cv_nse = 1 - cv_numerator / denominator
    
    return [cv_adjR2,cv_rmse,cv_nse,adjR2,rmse,nse,s,mae] 

def metricBetterThan( newMetric, oldMetric, perfMeasure ):
    """
    Function to compare two performance measure values and determine which one is more skillfull.
    """
    trueFalse = None

    if perfMeasure == "Adjusted R2" or perfMeasure == "Cross Validated Adjusted R2":
        if oldMetric > newMetric:
            trueFalse = False
        else:
            trueFalse = True

    elif perfMeasure == 'Root Mean Squared Error' or perfMeasure == 'Root Mean Squared Prediction Error' or perfMeasure == 'Mean Absolute Error':
        if oldMetric < newMetric:
            trueFalse = False
        else:
            trueFalse = True

    else:
        if oldMetric > newMetric:
            trueFalse = False
        else:
            trueFalse = True

    return trueFalse

def computeR2(yStar, yActual):
    yActual = np.array(yActual).flatten()
    yStar = np.array(yStar).flatten()
    yMean = float(np.mean(yActual))
    ssTotal = np.sum((yActual - yMean)**2)
    ssResidual = np.sum((yActual - yStar)**2)
    r2 = 1 - (ssResidual / ssTotal)
    return r2

def extract_nonexisting(archive):
    for name in archive.getnames():
        if os.path.exists(name):
            print(name, "already exists")
        else:
            archive.extract(name)
            print("extracting ", name)

#Set top-level path with xml enemble files to loop thru
mypath=("C:\Ensembles\HEFS")
os.chdir(mypath)
#Create empty dataframe for storing skill metrics
index = pd.date_range(start=pd.to_datetime('1985-01-01 '+'0:00:00'),end=pd.to_datetime('2012-09-01 '+'0:00:00'),freq='D')
cols=["Cross Validated Adjusted R2","Root Mean Squared Prediction Error","Cross Validated Nash-Sutcliffe",
"Adjusted R2","Root Mean Squared Error","Nash-Sutcliffe","Sample Variance","Mean Absolute Error"]
colnames=list(range(1985,2013))
colnames=list(map(str,colnames))
colnames.append('observed')
#metrics=pd.DataFrame(index=index,columns=cols)
fcst_vol=pd.DataFrame(index=index,columns=colnames)
# # #extract the files, if necessary
# for root, dirs, files in walk(mypath,topdown=True):
#     for f in files:
#         if f.endswith(".tgz"):
#             with tarfile.open(f) as archive:
#                 extract_nonexisting(archive)

#loop thru and gather pathnames
""" List of locations:
BBRW4   South Fork Shoshone River above Buffalo Bill Res
BHBW4   Bighorn River at Basin
BHRM8   Bighorn Lake inflows
CDYW4   Buffalo Bill Reservoir Inflows
CROW4   Wind River at Crowheart
DUBW4   Wind River at Dubois
GYBW4   Bighorn River at Greybull
KINW4   Wind River near Kinnear
LVEW4   Bighorn River near Kane
MEEW4   Greybull River at Meeteetse
NFSW4   North Fork River at Wapiti
RVTW4   Little Wind River near Riverton
SBDW4   Boysen Reservoir inflows
SLOW4   Shoshone River near Lovell
WDRW4   Wind River at Riverton
WRCW4   Wind River near Crowheart
N-Natural
blank-Depleted
"""
Station="BBRW4"
Stations=['BBRW4','BHBW4','BHRM8','CDYW4','CROW4','DUBW4','GYBW4','KINW4','LVEW4','MEEW4','NFSW4','RVTW4','SBDW4','SLOW4','WDRW4','WRCW4']
f = []
for root, dirs, files in walk(mypath,topdown=True):
    for x in files:
        if (x.endswith(".xml")) and (Station in x):
            f.append(os.path.join(root, x))
#Read csv of actual inflows to reservoirs
actual=pd.read_csv('C:\\Ensembles\\res_inflow_act.csv',parse_dates=True, infer_datetime_format = True, index_col=0)   
#initialize 3d array for storing metrics
#f=f[0:3]
l=len(f)
test=np.zeros((28,l,8))
#Loop thru, Open and parse xml
for count,file in enumerate(f):
    print("processing ", file)
    with open(file, 'rt') as document:
        #read in the document as a string
        docstring=document.read()
        #strip in faulty header language
        docstring=docstring.replace('<TimeSeries xmlns="http://www.wldelft.nl/fews/PI" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.wldelft.nl/fews/PI http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd" version="1.5">','<TimeSeries>' )
        #read string as xml tree
        tree = ElementTree.fromstring(docstring)
        #set column names from series indices
        columns=[]
        
        for mem_index in tree.iter('ensembleMemberIndex'):
            columns.append(str.strip(mem_index.text))
        testdate=pd.to_datetime('1985-01-01 '+'18:00:00')
        #pull the first instance to get start and end dates for pd df
        startDate=pd.to_datetime(tree.find('./series/header/startDate').attrib['date'] + ' ' + tree.find('./series/header/startDate').attrib['time'])
        endDate=pd.to_datetime(tree.find('./series/header/endDate').attrib['date'] + ' ' + tree.find('./series/header/endDate').attrib['time'])
        df = pd.DataFrame(index = pd.date_range(startDate, endDate,freq='6H'),columns=columns)
        
        members = []
        values = []
        timestamps = []

        #Loop through the series in the file
        for series in tree.findall('series'):
            locationId = series.find('./header/locationId')
            Member = series.find('./header/ensembleMemberIndex')
            members.append(Member.text.strip())
            #print(Member.text)
            #Loop through dates in the series and store in the pd dataframe
            values2 = []
            timestamps2 = []
            for event in series.iter('event'):
                date = event.attrib.get('date')
                time = event.attrib.get('time')
                #make value a float and assign cms units
                value = float(event.attrib.get('value'))*cms2cfs
                #print(value.to(ureg.cfs))
                #flag=event.attrib.get('flag')
                if date and time:
                    timestamps2.append(pd.to_datetime(date + ' ' + time))
                    values2.append(value)
                    #df.at[pd.to_datetime(date + ' ' + time),str.strip(Member.text)]=value
           
            values.append(values2)
            timestamps.append(timestamps2)
        
        
        # Members looks like: [MemberName1, MemberName2, ...]
        # values looks like: [
        #                       [mem1_value1, mem1_value2, ...],
        #                       [mem2_value1, mem2_value2, ...]
        #                    ]
        # Timestamps looks like values

        for k, member in enumerate(members):
            df2 = pd.DataFrame(data = values[k], index=pd.DatetimeIndex(timestamps[k]))
            df[member] = df2

        df=df.fillna('nearest')
        df=df.astype('float')
        df1=df.resample('D').mean()
        
        sd=df1.index[0]; ed=df1.index[-1]
        fcst_vol.at[sd,0:-1]=df1.sum()*cfsday2acft
        #store forecasted volumes for each trace, and observed
        yActual=actual[Station].loc[sd.date():ed.date()].to_numpy()
        fcst_vol.at[sd,'observed']=yActual.sum()*cfsday2acft
        #calculate continuous rank probability score
        
        #crps=ps.crps_ensemble(yActual.sum(), df1.sum().to_numpy())
        #calculate ranked histogram
        #ranky=rankz(yActual, df1.to_numpy())
        """ for i,column in enumerate(df1):
            yStar=df1[column].to_numpy()
            #store daily metrics
            test[i,count,:]=(computeMetrics(yStar, yStar, yActual, 0)) """
        """ if startDate==testdate:
            writer = pd.ExcelWriter('C:\\Ensembles\\HEFS\\test_ens5.xlsx', engine='xlsxwriter')
            actual['BHR'].loc[sd.date():ed.date()].to_excel(writer, sheet_name='volume')
            df.to_excel(writer, sheet_name='original')
            writer.save()
        """
OutputFile='C:\\Ensembles\\HEFS\\'+Station+'_metrics.xlsx'
writer = pd.ExcelWriter(OutputFile, engine='xlsxwriter')
fcst_vol.to_excel(writer, sheet_name='volume')
""" for i in range(test.shape[0]):
    df = pd.DataFrame(test[i,:,:])
    df.to_excel(writer, sheet_name='bin%d' % i) """
writer.save()
