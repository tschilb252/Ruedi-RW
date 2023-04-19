"""
Script:         ensemble_viewer.py
Author:         Jordan Lanini & Kevin Foley

Description:    
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os
from os import walk
from util import getHDB
from matplotlib import pyplot as plt, ticker as ticker
from pandas import DateOffset
import netCDF4
import datetime as dt
import pytz
maxlist = []
plt.rcParams['animation.convert_path'] = 'C:\\ImageMagick\\magick.exe'
NUM_ENSEMBLE_MEMBERS = 100
DATE_TICK_INDEX = [dt.strftime("%m") for dt in pd.date_range("1988-01-01", "1988-12-31")]
WEIGHTS = np.array([np.arange(121, 1, -1), np.arange(1, 121, 1)]).T
XTICKS = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
cfsday2acft = (24 * 60 * 60 / 43560)
cms2cfs = (3.2808 ** 3)


def InitializePlot():
    """
    Initializes the output plot
    with plot formatting options,
    initialized empty datasets,
    etc.
    """

    # Create the axes and the figure. Reduce fig size and dpi to make smaller files
    fig, ax = plt.subplots(
        figsize=(12, 6),
        dpi=300,
        facecolor="#fcfcfc"
    )

    # Set up the axes limits
    ax.set_xlim(left=1, right=366)
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
        x=0.01,
        y=.95,
        s="",
        fontsize=12,
        fontfamily='monospace'
    )
    # Set the location text placeholder
    locText = ax.text(
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='center',
        x=0.5,
        y=.95,
        s="",
        fontsize=12,
        fontfamily='monospace'
    )
    # Create the ensemble lines in the plot
    ensembleLines = []
    for i in range(NUM_ENSEMBLE_MEMBERS):
        ensembleLines.append(ax.plot(
            [], [],
            linewidth=0.25,
            color="#a6b064",
            label="_None"
        )[0])
    ensembleLines[0].set_label("Ensemble Member")

    # Create the Observation Line
    observationLine = ax.plot(
        [], [],
        linewidth=2,
        color="Black",
        label="Observation"
    )[0]
    # Create the median Line
    medianLine = ax.plot(
        [], [],
        linewidth=1.5,
        color="#3679ff",
        label="Median Forecast"
    )[0]
    # Add Legend
    ax.legend(frameon=True, loc='upper right', bbox_to_anchor=(1, 1), facecolor='w', fancybox=False, framealpha=1,
              edgecolor='w')

    # Create a vline to keep track of forecast date
    vl = ax.fill_betweenx([0, 0], 0, 0, color='r', lw=2, facecolor='b')

    return fig, ax, ensembleLines, observationLine, medianLine, dateText, locText, vl




def getUTForecastData(forecast_date='None'):
    """
    Returns an arbitrary number of days of
    ensemble data as a 2D array for the given year
    and julianDate (1-365).
    """
    df2 = df[df['initialization_time']==forecast_date] #mask to just the ensemble initialization date
    df2.index=pd.DatetimeIndex(pd.to_datetime(df2['valid_time'])).tz_convert('America/Denver')
    #df2.set_index(df2['valid_time'],inplace=True) #set the index and convert to local time

    #filter to just ensemble data
    df2=df2.loc[:,df2.columns.str.startswith('discharge_q')]
    start = dates[0].tz_convert('America/Denver') #january 1 of forecast year
    julianDays = (df2.index - start) / np.timedelta64(1, 'D') + start.dayofyear
    jdays = julianDays.to_numpy()
    ensembleData = df2.to_numpy()
    return jdays.flatten(), ensembleData

def num2date(seconds):
    delta=seconds * np.timedelta64(1, 's')
    date = dt.datetime(1970, 1, 1, tzinfo=pytz.timezone('utc'))+pd.to_timedelta(seconds,unit='s')
    return date

def getUTEnsembleData(array,date_array):
    """
    Returns an arbitrary number of days of
    ensemble data as a 2D array for the given year
    and julianDate (1-365).
    Reads netcdf file containing three dimensions: Initialization timestep, trace number, and flow rate.
    """
    dates=num2date(date_array) #convert from seconds to datetime
    dates=dates.tz_convert('America/Denver')  # january 1 of forecast year
    start = dt.datetime(dates[0].year,1,1, tzinfo=pytz.timezone('America/Denver'))
    julianDays = (dates - start) / np.timedelta64(1, 'D') + pd.to_datetime(start).dayofyear
    jdays = julianDays.to_numpy()
    ensembleData = np.array(array)
    return jdays.flatten(), ensembleData.transpose()


def update(i):
    """
    Update the animation with the new days
    ensembles and/or the new years
    observations.
    """
    forecast_date = dates[i]
    print(i)
    print(forecast_date)
    # ens = getUTEnsembleData(nc, forecast_date)
    # Figure out the date
    julianDate = forecast_date.dayofyear
    # Update the observations if julianDate = 0 (i.e. a new year)
    if ENSEMBLE==True:
        ensembleDates, ensembleData = getUTEnsembleData(nc['Flow'][i+icount, :, :], nc['timestep'][i+icount, :])
    else:
        ensembleDates, ensembleData = getUTForecastData(forecast_date)
    if i == 0:
        mask = (obs.index.tz_localize('America/Denver') >= dates[0]) & (
                    obs.index.tz_localize('America/Denver') <= (dates[-1] + DateOffset(days=366)))
        obsdata = obs.loc[mask]  # mask out the observations for the plot
        julianDays = (obsdata.index - obsdata.index[0]) / np.timedelta64(1, 'D') + obsdata.index[0].dayofyear
        observationLine.set_data(julianDays, np.nan_to_num(obsdata.to_numpy()))
        locText.set_text(location)
        top = max(np.array(obsdata.max()) * 1.10, ensembleData.max() * 1.10)
        ax.set_ylim(bottom=0, top=top)  # set ylimits
        ax.set_xlim(left=dates[0].dayofyear, right=dates[-1].dayofyear + 10)
    # set the date text for displaying in the graph
    dateText.set_text(forecast_date.strftime("%b %d %Y"))
    # Get the data for this particular date
    # ensembleDates, ensembleData = getUTForecastData(forecast_date)


    # Update vline
    ax.collections.clear()
    vl = ax.fill_betweenx([0, np.average(maxlist)], 0, julianDate, color='k', alpha=0.1, hatch="////")

    # Set the ensemble member data
    for count, member in enumerate(ensembleLines):
        member.set_data(ensembleDates, ensembleData[:, count])
    medianLine.set_data(ensembleDates, np.median(ensembleData, axis=1))
    return observationLine, ensembleLines, medianLine

if __name__ == '__main__':
    NATURAL = True #Process natural inflow forecasts (True) or Actual (False)
    ENSEMBLE= False #Plot ensemble or original forecast uncertainty bounds

    if NATURAL:
        sdi="100840" #SDI for data retreival from HDB
        fname="HydroForecast_short-term_doe-ruedi-reservoir_Natural_Flows.csv"
        location = 'Ruedi Reservoir Natural Inflow' #set the location name in the figure
    else:
        sdi = "101024"  # SDI for data retreival from HDB
        fname = "HydroForecast_short-term_doe-ruedi-reservoir_Actual_Flows.csv"  # filename to read forecast
        location = 'Ruedi Reservoir Actual Inflow'
        # Set top-level path with csv ensemble files to loop thru and output directory
    mypath = ("C:\\Users\\JLanini\\OneDrive - DOI\\Documents\\GitHub\\Ruedi-RW\\Data\\UpstreamTech\\")
    outdir = (mypath+"\\Figures\\")
    os.chdir(mypath)
    fp=mypath + "Ensembles\\Ruedi_Ens.nc"
    # read in pre-saved observations
    obs = getHDB(sdi=sdi, sd=pd.to_datetime('2010-01-01'), ed=pd.to_datetime('2022-12-31'))  # pull in observed values
    #obs.index = obs.index.tz_convert('America/Denver')
    #obs = obs.resample('H').interpolate(method='polynomial', order=3)  # resample to hourly

    if not(ENSEMBLE):
        data=pd.read_csv(fname) #original data with confidence intervals
        # initialize lists for files and dates, used to sort file list
        data['initialization_time']=pd.to_datetime(data['initialization_time'])
        data['valid_time'] = pd.to_datetime(data['valid_time'])
        alldates = data['initialization_time'].unique()
        NUM_ENSEMBLE_MEMBERS = 10
    if ENSEMBLE:
        nc = netCDF4.Dataset(fp)
        NUM_ENSEMBLE_MEMBERS = 100
        alldates=num2date(nc['time'][:])
    fig, ax, ensembleLines, observationLine, medianLine, dateText, locText, vl = InitializePlot()
        #loop through the range of years in the files
    icount=0 #counter for how many frames have been processed
    for year in range(alldates.year.min(), alldates.year.max()+1):
        #run the animation.  The number of frames sets the time period that we will display.  Because we have
        #weekly data, the frames are calculated as the number of days divided by 7.
        print("Creating animation for "+str(year))
        sd=pd.to_datetime(str(year)+'-01-01').tz_localize('utc')
        ed=pd.to_datetime(str(year)+'-12-31').tz_localize('utc')
        if not (ENSEMBLE):
            mask=(data['initialization_time']>=sd) & (data['initialization_time']<=ed)
            df=data.loc[mask]
            dates = df['initialization_time'].unique()
        else:
            mask = (alldates >= sd) & (alldates <= ed) #mask dates to year selected
            dates=alldates[mask]
        ani = animation.FuncAnimation(fig, update, frames=len(dates), interval=10)
        # Set gif writer and save to file
        writer = animation.ImageMagickFileWriter(fps=7)
        ani.save(outdir +"\\animations\\"+ str(year) + '_anim.gif', writer=writer)
        icount=icount+len(dates)
        #save individual frames
        #print("Saving frames for " + str(year))
        #for frame in range(0,int((365 - 122) / 7)):
        #    update(frame)
        #    plt.savefig(outdir +"\\frames\\"+ str(year) +'_'+ str(frame)+'.png')





