import datetime
import pandas as pd
import numpy as np
import datetime as dt
import netCDF4
import os
import pickle
import pytz
from matplotlib import pyplot as plt
import matplotlib as mpl
from math import floor,ceil

from pandas import DateOffset


def CreateNetCDF(data_path, filename, n_traces, n_tsteps, n_forecasts):
    # Create NetCDF File
    output_nc = os.path.join(data_path, filename)
    nc = netCDF4.Dataset(output_nc, 'w')
    # Global attributes
    nc.title = 'UpstreamTech Ensemble'
    nc.summary = ('Ensemble Traces created from an UpstreamTech 10-day median forecast for Ruedi Reservoir'
                  'for 7/2015-9/2021')
    nc.institution = 'Bureau of Reclamation Technical Service Center'
    nc.history = '{0} creation of ensemble netcdf file.'.format(
        dt.datetime.now().strftime("%Y-%m-%d")
    )
    trace_dim = nc.createDimension('trace', n_traces)
    fcst_dim = nc.createDimension('init_time', n_forecasts)
    tstep_dim = nc.createDimension('timestep', n_tsteps)
    # Create variables
    trace_var = nc.createVariable('trace', np.int8, ('trace'))
    trace_var.units = 'count'
    trace_var.standard_name = 'trace number'
    trace_var.axis = 'Y'

    time_var = nc.createVariable('time', np.int32, ('init_time'))
    time_var.standard_name = 'timestep'
    time_var.calendar = 'gregorian'
    time_var.time_step = 'Monthly'
    time_var.units = 'Seconds since 1970-01-01 00:00:00'
    time_var.axis = 'T1'

    timestep_var = nc.createVariable('timestep', np.int32, ('init_time', 'timestep'))
    timestep_var.standard_name = 'timestep'
    timestep_var.calendar = 'gregorian'
    timestep_var.time_step = 'Monthly'
    timestep_var.units = 'Seconds since 1970-01-01 00:00:00'
    timestep_var.axis = 'T2'

    flow_var = nc.createVariable('Flow', np.float64, ('init_time', 'trace', 'timestep'),
                                 fill_value=np.nan)
    flow_var.units = 'CFS'
    flow_var.long_name = 'Cubic Feet per Second'
    flow_var.short_name = 'CFS'
    return nc, time_var, trace_var, timestep_var, flow_var


def date2num(date):
    delta = (date.tz_convert('UTC') - dt.datetime(1970, 1, 1, tzinfo=pytz.timezone('utc')))
    seconds = delta / np.timedelta64(1, 's')
    return np.int32(seconds)


def getUTEnsembleData(forecast_date='None', df=pd.DataFrame()):
    """
    Returns an arbitrary number of days of
    ensemble data as a 2D array for the given year
    and julianDate (1-365).
    """
    # df.index = pd.DatetimeIndex(pd.to_datetime(df['valid_time'])).tz_convert('America/Denver')
    df2 = df[df['initialization_time'] == forecast_date]  # mask to just the ensemble initialization date
    df2.index = pd.DatetimeIndex(pd.to_datetime(df2['valid_time'])).tz_convert('America/Denver')
    # df2.set_index(df2['valid_time'],inplace=True) #set the index and convert to local time

    # filter to just ensemble data
    df2 = df2.loc[:, df2.columns.str.startswith('discharge_q')]
    return df2


def RandomWalk(dims=1, step_n=241,range=999,init_error=0.1):
    # Define parameters for the walk
    step_set = [-range/step_n, range/step_n] #make the step set so it can reach outer bounds by the end of the walk
    origin = np.zeros((1, dims))
    origin.fill((0.5+((np.random.rand()-0.5)*init_error))*range)
    # Simulate steps in 1D
    step_shape = (step_n - 1, dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    #steps = np.random.exponential(scale=1, size=step_shape) * np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    #path = (path - path.min()) / (path.max() - path.min())  # normalize to 0-1
    return path/range
def flatten(l):
    return [sublist[1] for sublist in l]

ENS_TRACES = 100
window = 15  # number of days to window for exceedence calculation
path = ("C:\\Users\\JLanini\\OneDrive - DOI\\Documents\\GitHub\\Ruedi-RW\\Data\\UpstreamTech\\")
NATURAL = True  # Process natural inflow forecasts (True) or Actual (False)
if __name__ == '__main__':
    os.chdir(path)
    if NATURAL:
        sdi = "100840"  # SDI for data retreival from HDB
        #fname = "HydroForecast_short-term_doe-ruedi-reservoir_Natural_Flows.csv"
        fname = "HydroForecast_model-v2_short-term_doe-ruedi-reservoir_Natural_Flows.csv"
        location = 'Ruedi Reservoir Natural Inflow'  # set the location name in the figure
    else:
        sdi = "101023"  # SDI for data retreival from HDB
        fname = "HydroForecast_short-term_doe-ruedi-reservoir_Actual_Flows.csv"  # filename to read forecast
        location = 'Ruedi Reservoir Actual Inflow'
        # Set top-level path with csv ensemble files to loop thru and output directory
    mypath = ("C:\\Users\\JLanini\\OneDrive - DOI\\Documents\\GitHub\\Ruedi-RW\\Data\\UpstreamTech\\")
    data = pd.read_csv(mypath + fname)
    # initialize lists for files and dates, used to sort file list
    data['initialization_time'] = pd.to_datetime(data['initialization_time'])
    data['valid_time'] = pd.to_datetime(data['valid_time'])
    alldates = data['initialization_time'].unique()
    start = pd.to_datetime(pd.to_datetime(alldates[0]))
    end = pd.to_datetime(pd.to_datetime(alldates[-1]))
    # pull in observed values
    # obs = getHDB(sdi=sdi, tstp='HR',sd=start - DateOffset(days=10), ed=end + DateOffset(days=10))
    # obs.to_csv(mypath+"obs.csv")
    obs = pd.read_csv(mypath + "obs.csv")  # read back in locally stored observations
    obs['t'] = pd.to_datetime(obs['t'])
    obs.set_index('t', inplace=True)
    obs.index = obs.index.tz_localize('America/Denver')
    obs = obs.resample('H').interpolate(method='polynomial', order=3)  # resample to hourly
    dr1 = pd.date_range(pd.to_datetime(datetime.datetime(2020, 1, 1)),
                        pd.to_datetime(datetime.datetime(2020, 12, 31)),
                        freq='D')  # set up a daily timestep loop for one year
    #########################################Calculate errors from dataset##############################################
    # error = np.empty((366, (int(end.year - start.year)+1) * window, 241))  # loop thru each day of the year
    # error[:,:,:] = np.nan  # reset list of errors
    # for doy, dt in enumerate(dr1):
    #     print("processing " + str(dt.dayofyear))
    #     count = 0
    #     for j, year in enumerate(range(start.year, end.year+1)):
    #         if ( (not start.is_leap_year) & (dt.month == 2) & (dt.day == 29)):
    #             day = 28
    #         else:
    #             day = dt.day
    #         sd = pd.to_datetime(datetime.datetime(year, dt.month, day)) - DateOffset(
    #             days=floor(window / 2))  # window start timestep
    #
    #         ed = pd.to_datetime(datetime.datetime(year, dt.month, day)) + DateOffset(
    #             days=ceil(window / 2))  # window end timestep
    #         mask = (alldates >= sd.tz_localize('America/Denver')) & (alldates <= ed.tz_localize('America/Denver'))  #
    #         if (len(alldates[mask]) == window):
    #             # start calculations when we have a full dataset
    #             for i, dt2 in enumerate(alldates[mask]):
    #                 forecast = getUTEnsembleData(dt2, data)[
    #                     'discharge_q0.5']  # .resample('D').mean()  # pull out the ensemble in question
    #                 error[doy, count, :] = (forecast.values - obs[
    #                     (obs.index >= forecast.index[0]) & (obs.index <= forecast.index[-1])].values.flatten())
    #                 print("year "+str(year)+"window day " + str(count))
    #                 count = count + 1
    # # create ensemble
    #Plot monthly error exceedence curves##############################################################################
    #with open(mypath+'errors.pkl', 'wb') as f: pickle.dump(error, f)
    with open('errors.pkl', 'rb') as f:
        error = pickle.load(f)  # np array: error[Julian Day of Year,values,timestep]
    fig, axs = plt.subplots(3, 4, figsize=(10, 6))
    color = mpl.cm.Oranges(np.linspace(0, 1, 10))
    dr2 = pd.date_range(pd.to_datetime(datetime.datetime(2020, 1, 1)),
                        pd.to_datetime(datetime.datetime(2020, 12, 31)),
                        freq='M')  # set up a Monthly timestep loop for one year
    for lead_step in range(0, 241): #loop thru forecast lead time, in hours
        exceed = np.empty((3, 4, 10000))  # create 3d array
        exceed[:] = np.nan  # initialize to nans
        for day in dr1: #loop through day of years
            row = int((day.month - 1) // 4) #calculate plot row
            col = int((day.month - 1) % 4) #calculate plot column
            vals=np.append(exceed[row,col,:],error[day.dayofyear - 1, :, lead_step])
            vals = vals[~np.isnan(vals)]  # drop nans
            idx = range(len(vals))
            exceed[row,col,idx]=vals #append and expand array
            #exceed[row,col]=exceed[row,col][~np.isnan(exceed[row,col])] #drop nans
        for month in dr2:
            row = int((month.month - 1) // 4)
            col = int((month.month - 1) % 4)
            vals = exceed[row, col, :]
            vals = vals[~np.isnan(vals)]  # drop nans
            size=1
            if row==1 and col==0 and lead_step==0:
                axs[row, col].plot(lead_step,np.percentile(vals,75), 'b.',label='25%-75% Exceedence',ms=size)
                axs[row, col].plot(lead_step, np.percentile(vals, 95), 'r.', label='5%-95% Exceedence', ms=size)
                axs[row, col].plot(lead_step, np.percentile(vals, 50), 'g.', label='Median', ms=size)
            else:
                axs[row, col].plot(lead_step, np.percentile(vals, 75), 'b.', ms=size)
                axs[row, col].plot(lead_step, np.percentile(vals, 95), 'r.', ms=size)
                axs[row, col].plot(lead_step, np.percentile(vals, 50), 'g.', ms=size)
            axs[row, col].plot(lead_step, np.percentile(vals,25), 'b.',ms=size)
            axs[row, col].plot(lead_step, np.percentile(vals,5),'r.',ms=size)

    for month in dr2:
        row = int((month.month - 1) // 4)
        col = int((month.month - 1) % 4)
        if col==0:
            axs[row, col].set_ylabel("Forecast-Observation, cfs")
        if row == 2:
            axs[row, col].set_xlabel("Lead Time, hours")
        axs[row, col].set_title(month.strftime("%B"))

    # fig.colorbar(axs[row,col])
    fig.tight_layout(pad=1.0)
    fig.legend()
    plt.savefig(mypath + 'errorcone.png')
    plt.show()
    plt.close(fig)

    fig, axs = plt.subplots(3, 4, figsize=(10, 6))
    color = mpl.cm.Oranges(np.linspace(0, 1, 10))
    for lead_step, c in zip(range(0, 241), color):
        for day in dr1:
            row = int((day.month - 1) // 4)
            col = int((day.month - 1) % 4)
            print(str(row), str(col))
            axs[row, col].plot(np.sort(error[day.dayofyear - 1, :, lead_step]), c=c)
        dr2 = pd.date_range(pd.to_datetime(datetime.datetime(2020, 1, 1)),
                            pd.to_datetime(datetime.datetime(2020, 12, 31)),
                            freq='M')  # set up a Monthly timestep loop for one year
        for month in dr2:
            row = int((month.month - 1) // 4)
            col = int((month.month - 1) % 4)
            axs[row, col].set_title(month.strftime("%B"))

    # fig.colorbar(axs[row,col])
    fig.tight_layout(pad=1.0)
    plt.savefig(mypath + 'errordist.png')
    #Create netcdf for output
    # nc, time_var, trace_var, timestep_var, flow_var = CreateNetCDF(data_path=path + "Ensembles\\",
    #                                                                filename="Ruedi_Ens.nc", n_traces=ENS_TRACES,
    #                                                                n_tsteps=241,
    #                                                                n_forecasts=len(alldates))
    # time_var[:] = date2num(alldates)  # load initialization times to netcdf file
    # trace_var[:] = range(0, ENS_TRACES)  # load trace numbers to netcdf file
    # for t,forecast_day in enumerate(alldates):
    #     print(forecast_day.strftime('%Y-%m-%d'))
    #     error = getUTEnsembleData(forecast_day, data)
    #     colnums=pd.DataFrame(error.columns).iloc[:,0].str.split('discharge_q')
    #     colnums=flatten(colnums)
    #     test=pd.DataFrame(columns=range(1,1000,1))#get the full distribution
    #     for i,num in enumerate(colnums):
    #         test.loc[:,int(float(num)*1000)]=error.iloc[:,i]
    #     for col in test:
    #         test[col] = pd.to_numeric(test[col], errors='coerce')
    #     error_dist=test.interpolate(method='linear', axis=1).fillna(method='bfill',axis=1)
    #     #error_dist.iloc[0, :].plot()
    #     forecast = getUTEnsembleData(forecast_day, data)['discharge_q0.5']  # get median forecast from the upstream tech forecast
    #     cols = range(0, ENS_TRACES)
    #     cols = ["trace " + str(x) for x in cols]
    #     timestep_var[t, :] = date2num(forecast.index)
    #     ensemble = pd.DataFrame(data=np.nan, index=forecast.index, columns=cols)  # create empty dataframe for ensembles
    #     walk_ensemble = pd.DataFrame(data=np.nan, index=forecast.index,
    #                                  columns=cols)  # create empty dataframe for ensembles
    #     trace_obs = obs[
    #         (obs.index >= forecast.index[0]) & (obs.index <= forecast.index[-1])]  # window out the observations
    #     for trace in range(0, ENS_TRACES):
    #         rand_array = RandomWalk()  # get a random walk to pull exceedences from
    #         walk_ensemble.iloc[:, trace] = rand_array
    #         for tstep,step_dt in enumerate(forecast.index):  # loop thru the timesteps
    #             dist = np.sort(error_dist.loc[step_dt, :])
    #             ensemble.iloc[tstep, trace] = dist[int(np.floor(rand_array[tstep] * (len(dist) - 1)))]
    #         flow_var[t, trace, :] = np.array(ensemble.iloc[:,trace])
    #     # ensemble.to_csv(path + "Ensembles\\" + forecast_day.strftime('%Y-%m-%d') + ".csv")
    # nc.close()
