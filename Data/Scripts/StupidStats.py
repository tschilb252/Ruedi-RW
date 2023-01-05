"""
Script:         not_an_ensemble_viewer.py
Author:         Jordan Lanini & Kevin Foley

Description:    
"""
# import flask
# from flask import Flask

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import hydrostats
import HydroErr as he
from datetime import datetime
import pandas as pd
import numpy as np
import os
from os import walk
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from pandas.tseries.offsets import DateOffset

#app = Flask(__name__)

# """ @app.route("/")
# def home():
#     return "Hello, Flask!"

# @app.route("/hello/<name>")

# def hello_there(name):
#     maxlist=[] """
plt.rcParams['animation.convert_path'] = 'C:\\ImageMagick\\magick.exe'

DATE_TICK_INDEX = [dt.strftime("%m") for dt in pd.date_range("1988-01-01", "1989-04-30")]
WEIGHTS = np.array([np.arange(121, 1, -1), np.arange(1, 121, 1)]).T
XTICKS = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335,366,397,425,456,486]
mypath=("C:\\Ensembles\\Deterministic Runs\\Data")
os.chdir(mypath)
files=[]

cfsday2acft=(24*60*60/43560)
cms2cfs=(3.2808**3)
def bias(Y_actual,Y_Predicted):
 
    op=pd.concat([Y_actual,Y_Predicted],axis=1)
    s.rename()
    op=op.dropna()
    op.rename(columns={'data':'actual','Y_Predicted':'predicted'},inplace=True)
    #df=Y_actual.merge(Y_Predicted, left_index=True)
    percent_bias = np.mean((op['actual'] - op['predicted'])/op['actual'])*100
    return percent_bias
sta_dict={
    'BBRW4':'SFS',
    'BHBW4':'BBWY',
    'CROW4':'WRCH', 
    'DUBW4':'WRND', 
    'GYBW4':'BHRW', #using worland as a proxy for now
    'KINW4':'WRKY', 
    'LVEW4':'KAWY', 
    'MEEW4':'GRMY', 
    'NFSW4':'NFS', 
    'RVTW4':'LWRV', 
    'SLOW4':'SRLY', 
    'VLYW4':'SFVY', 
    'WDRW4':'WRRY', 
    'WRCW4':'WRDY', 
    'BLRW4':'BLR',
    'CDYW4':'BBR', 
    'SBDW4':'BOYR', 
    'BHRM8':'BHR'}
parm_dict={
    'SFS':'QD',
    'BBWY':'QD',
    'WRCH':'QD', 
    'WRND':'QD', 
    'BHRW':'QD', #using worland as a proxy for now
    'WRKY':'QD', 
    'KAWY':'QD', 
    'GRMY':'QD', 
    'NFS':'QD', 
    'LWRV':'QD', 
    'SRLY':'QD', 
    'SFVY':'QD', 
    'WRRY':'QD', 
    'WRDY':'QD', 
    'BLR':'IN',
    'BBR':'IN', 
    'BOYR':'IN', 
    'BHR':'IN'}
def InitializePlot():
    """
    Initializes the output plot
    with plot formatting options,
    initialized empty datasets, 
    etc.
    """

    # Create the axes and the figure. Reduce fig size and dpi to make smaller files
    fig, ax = plt.subplots(
        figsize=(10,5), 
        dpi=50, 
        facecolor="#fcfcfc"
        )

    # Set up the axes limits
    ax.set_xlim(left=1, right=366+122)
    ax.set_ylim(bottom=0, top=15000)

    # Set Axes Formatter
    ax.xaxis.set_major_formatter(ticker.IndexFormatter(DATE_TICK_INDEX))
    ax.set_xticks(XTICKS)

    # Set Grid Lines
    ax.grid(which="both", color="#b5b5b5")

    # Set Axes Titles
    ax.set_xlabel("Forecast Date")
    ax.set_ylabel("Discharge [cfs]")

    # Set the year text placeholder
    dateText = ax.text(
        transform=ax.transAxes,
        verticalalignment='bottom',
        x = 0.01, 
        y = .95,
        s = "",
        fontsize = 12,
        fontfamily = 'monospace'
    )
    # Set the location text placeholder
    locText = ax.text(
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='center',
        x = 0.5, 
        y = .95,
        s = "",
        fontsize = 12,
        fontfamily = 'monospace'
    )
    # Create the ensemble lines in the plot
    ensembleLines = []
    for i in range(NUM_ENSEMBLE_MEMBERS):
        ensembleLines.append(ax.plot(
            [],[],
            linewidth = 0.25,
            color = "#a6b064",
            label = "_None"
            )[0])
    ensembleLines[0].set_label("Ensemble Member")
    
    # Create the Observation Line
    observationLine = ax.plot(
        [],[],
        linewidth = 2,
        color = "#3679ff",
        label = "Observation"
    )[0]

    # Add Legend
    ax.legend(frameon=True, loc='upper right', bbox_to_anchor=(1,1), facecolor='w', fancybox=False, framealpha=1, edgecolor='w')

    # Create a vline to keep track of forecast date
    vl = ax.fill_betweenx([0,0], 0, 0, color='r', lw=2, facecolor='b')

    return fig, ax, ensembleLines, observationLine, dateText,locText, vl

def parse_xml_line(line, key):
    """
    finds the value associated with 'key'
    in the provided cml-formatted 'line'
    """
    data = line[line.index(key+'='):]
    data = data[data.index('=')+2:]
    data = data[:data.index('"')]
    return data

def get_inner_xml(line):
    """
    Gets the inner XML of the line
    """
    return line[line.index(">")+1:line.index("</")]

def get_file_names(mydir):
    #gets a list of the xml filenames in the directory for a particular station
    f=[]
    for root, dirs, files in walk(mydir,topdown=True):
        for x in files:
            if x.endswith(".xml"):
                f.append(os.path.join(root, x))
    return f
def calc_actual(sd,ed,Station):
    yActual=actual[Station].loc[sd.date():ed.date()].to_numpy()
    return(yActual.sum()*cfsday2acft)
def get_actual(sd,ed,Station):
    #Gets observed values for a period
    yActual=actual[Station].loc[sd.date():ed.date()]
    return(yActual.sum()*cfsday2acft)
def process_det_file(file_name):

    """
    Reads in the file and creates a dataframe containing
    the ensembles from that file:
    """

    headerCounter = -1
    forecast_date = None
    location_id = None

    # Open the file and read in the lines
    with open(file_name, 'r') as readfile:
        file_lines = readfile.readlines()

    # Create arrays to store data
    location_id = []
    data_values = []
    data_timestamps = []

    # Iterate over the lines and collect data
    for line in file_lines:

        # Figure out when we're in the header
        if '<header' in line:
            headerCounter = headerCounter + 1
            data_values.append([])

        # Get the forecast issue date
        if 'forecastDate' in line:
            forecast_date = parse_xml_line(line, 'date')
        
        # Get the location ID
        if 'locationId' in line:
            location_id.append(get_inner_xml(line))
        # Get the Start Date
        if 'startDate' in line:
            sd = pd.to_datetime(parse_xml_line(line, 'date'))
        # Get the Start Date
        if 'endDate' in line:
            ed = pd.to_datetime(parse_xml_line(line, 'date'))
        
        # Add data as it comes in
        if 'event' in line:
            date = parse_xml_line(line, 'date')
            time = parse_xml_line(line, 'time')
            data_values[headerCounter].append(parse_xml_line(line, 'value'))
            if headerCounter < 1:
                data_timestamps.append(date + ' ' + time)
                
        
    # Create a dataframe
    #ensemble_members.append('Observed')
    data = np.array(data_values)
    data = data.T
    df = pd.DataFrame(columns=location_id, data = data, index = pd.DatetimeIndex(map(pd.to_datetime, data_timestamps)))
    
    df = df.astype('float')
    #df = df*cms2cfs
    
    df = df.resample("D").mean()
    return df
def process_ens_file(file_name,Station):
    
    """
    Reads in the file and creates a dataframe containing
    the ensembles from that file:
    """

    headerCounter = -1
    forecast_date = None
    location_id = None

    # Open the file and read in the lines
    with open(file_name, 'r') as readfile:
        file_lines = readfile.readlines()

    # Create arrays to store data
    ensemble_members = []
    data_values = []
    data_timestamps = []

    # Iterate over the lines and collect data
    for line in file_lines:

        # Figure out when we're in the header
        if '<header' in line:
            headerCounter = headerCounter + 1
            data_values.append([])

        # Get the forecast issue date
        if 'forecastDate' in line and headerCounter < 1:
            forecast_date = parse_xml_line(line, 'date')
        
        # Get the location ID
        if 'locationID' in line and headerCounter < 1:
            location_id = get_inner_xml(line)
        # Get the Start Date
        if 'startDate' in line and headerCounter < 1:
            sd = pd.to_datetime(parse_xml_line(line, 'date'))
        # Get the Start Date
        if 'endDate' in line and headerCounter < 1:
            ed = pd.to_datetime(parse_xml_line(line, 'date'))
        # Get the ensemble member year
        if 'ensembleMemberIndex' in line:
            ensemble_members.append(get_inner_xml(line))
        
        # Add data as it comes in
        if 'event' in line:
            date = parse_xml_line(line, 'date')
            time = parse_xml_line(line, 'time')
            data_values[headerCounter].append(parse_xml_line(line, 'value'))
            if headerCounter < 1:
                data_timestamps.append(date + ' ' + time)
                
        
    # Create a dataframe
    #ensemble_members.append('Observed')
    data = np.array(data_values)
    data = data.T
    df = pd.DataFrame(data = data, index = pd.DatetimeIndex(map(pd.to_datetime, data_timestamps)), columns=ensemble_members)
    
    df = df.astype('float')
    df = df*cms2cfs
    
    df = df.resample("D").mean()
    
    Obs=actual[Station].loc[pd.to_datetime('1985-01-01 '+'0:00:00'):ed.date()]
    return df,Obs
def getObservations(year = None,location='BHRM8'):
    """
    Returns 1-year of observations as a 
    1D numpy array. Takes the year as an
    input and returns that years data.
    """
    #set index to January 1 of year to the end of the year plus the length of an ensemble
    index = pd.date_range(start=pd.to_datetime(str(year)+'-01-01 '+'0:00:00'),end=pd.to_datetime(str(year)+'-01-01 '+'0:00:00')+DateOffset(days=366+122),freq='D')
    observations = actual[location].loc[index].to_numpy()

    return observations


def getEnsembleData(file_='None',location='BHRM8'):
    """
    Returns an arbitrary number of days of
    ensemble data as a 2D array for the given year
    and julianDate (1-365).
    """
    df,Obs=process_file(file_,location)
    start=pd.to_datetime(df.index[0].strftime('%Y')+'-01-01')
    julianDays = (df.index - start)/np.timedelta64(1,'D')+1
    jdays=julianDays.to_numpy()
    ensembleData = df.to_numpy()
    """ for i,ens in enumerate(df):
    ensembleData[:, i] = ens """

    return jdays.flatten(), ensembleData

def getHydromet(sta,parm,sd,ed):
    URL = 'https://www.usbr.gov/gp-bin/webarccsv.pl?parameter='+sta+'%20'+parm+'&syer='+str(sd.year)+'&smnth='+str(sd.month)+'&sdy='+str(sd.day)+'&eyer='+str(ed.year)+'&emnth='+str(ed.month)+'&edy='+str(ed.day)+'&format=4'
    df = pd.read_csv(URL,header=0,index_col=0,parse_dates=True,converters={"data":lambda x: pd.to_numeric(x, errors='coerce')}, names=['data'])
    return df

if __name__ == '__main__':
    #obs=getHydromet('BBR','IN',pd.to_datetime('1900-01-01'),pd.to_datetime('2020-01-01'))
    #read model data file for depleted flows
    df=process_det_file('Bighorn_HistSim_2.xml')
    for location in sta_dict.keys():
        if location == 'CDYW4':
            pred=df[location]
            obs=getHydromet(sta_dict[location],parm_dict[sta_dict[location]],pred.index.min(),pred.index.max())
            #obs.plot()
            #plt.show()
            #op=pd.concat()
            mae=he.mae(pred.values.flatten(),obs.values.flatten())
            nse=he.nse_mod(pred.values.flatten(),obs.values.flatten())
            perbias=bias(obs,pred)
            print("NSE=",nse,"MAE=",mae)
            #mae=he.rmse(np.array(pred),np.array(obs))