"""
Script:         ensemble_viewer.py
Author:         Kevin Foley

Description:    
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from datetime import datetime
import pandas as pd
import numpy as np
import os
from os import walk
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from pandas.tseries.offsets import DateOffset
plt.rcParams['animation.convert_path'] = 'C:\\ImageMagick\\magick.exe'
actual=pd.read_csv('C:\\Ensembles\\res_inflow_act.csv',parse_dates=True, infer_datetime_format = True, index_col=0)   
NUM_ENSEMBLE_MEMBERS = 28
DATE_TICK_INDEX = [dt.strftime("%b-%d") for dt in pd.date_range("1988-01-01", "1989-03-31")]
WEIGHTS = np.array([np.arange(121, 1, -1), np.arange(1, 121, 1)]).T
XTICKS = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335,365,396,424,455]
mypath=("C:\Ensembles\HEFS")
os.chdir(mypath)
files=[]
cfsday2acft=(24*60*60/43560)
cms2cfs=(3.2808**3)
def InitializePlot():
    """
    Initializes the output plot
    with plot formatting options,
    initialized empty datasets, 
    etc.
    """

    # Create the axes and the figure
    fig, ax = plt.subplots(
        figsize=(10,5), 
        dpi=100, 
        facecolor="#fcfcfc"
        )

    # Set up the axes limits
    ax.set_xlim(left=1, right=366+122)
    ax.set_ylim(bottom=0, top=300)

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
    yActual=actual[Station].loc[sd.date():ed.date()]
    return(yActual.sum()*cfsday2acft)
def process_file(file_name,Station):
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
    index = pd.date_range(start=pd.to_datetime(str(year)+'-01-01 '+'0:00:00'),end=pd.to_datetime(str(year)+'-01-01 '+'0:00:00')+DateOffset(days=366+122),freq='D')
    # Make up some toy data
    
    observations = actual[location].loc[index]

    return observations


def getEnsembleData(file_='None',location='BHRM8'):
    """
    Returns an arbitrary number of days of
    ensemble data as a 2D array for the given year
    and julianDate (1-365).
    """
    fname=file_.split('\\')[-1]
    
    df,Obs=process_file(file_,location)
    # Generate some toy data
    julianDays = (df.index - df.index[0])/np.timedelta64(1,'D')+1
    jdays=julianDays.to_numpy()
    ensembleData = df.to_numpy()
    """ for i,ens in enumerate(df):
       ensembleData[:, i] = ens """

    return jdays, ensembleData



if __name__ == '__main__':

    # Set up the figure, axes, and lines
    fig, ax, ensembleLines, observationLine, dateText,locText, vl = InitializePlot()
    # Iterate over forecast points
    location = 'LVEW4'#'NFSW4']#'BBRW4']#,'BHBW4','BHRM8','CDYW4','CROW4','DUBW4','GYBW4','KINW4','LVEW4','MEEW4','NFSW4','RVTW4','SBDW4','SLOW4','WDRW4','WRCW4']
    files = get_file_names(mypath)
    f = list(filter(lambda filename: location in filename, files))
        # Create the animation
    # Animation Update Function
    def update(i):
        """
        Update the animation with the new days
        ensembles and/or the new years
        observations.
        """
        file_=f[i]
        fname=file_.split('\\')[-1]
        forecast_date = pd.to_datetime(fname[:8], format='%Y%m%d')
        year = forecast_date.year
        # Figure out the date
        #year = int(frame/365) + 1985
        julianDate = forecast_date.dayofyear
        files = get_file_names(mypath)
        # Update the observations if julianDate = 0 (i.e. a new year)
        if julianDate == 1:
            observationLine.set_data(list(range(1,366+122)), getObservations(year=year,location=location))
            locText.set_text(location)
        dateText.set_text(forecast_date.strftime("%b %d %Y"))
        # Get the data for this particular date
        ensembleDates, ensembleData = getEnsembleData(file_, location)

        # Set the y axis range
        max_ = np.max([ensembleData.max(), max(observationLine._yorig)])
        mag = int(np.log10(max_))
        top = max_ + ((10**mag) - max_%(10**mag))
        ax.set_ylim(bottom=0, top=top)

        # Update vline
        ax.collections.clear()
        vl = ax.fill_betweenx([0, top], 0, julianDate, color='k', alpha=0.1, hatch="////")

        # Set the ensemble member data
        for i, member in enumerate(ensembleLines):
            member.set_data(ensembleDates, ensembleData[:,i])

        return ensembleLines,
    
    ani = animation.FuncAnimation(fig, update, frames=365)
    writer = animation.ImageMagickFileWriter()
    #ani.save(location+'_anim.gif', writer=writer)
    plt.show()