import json
import os
import urllib.request
from os import walk
from textwrap import wrap

import pandas as pd
import seaborn as sns
from bokeh.io import curdoc, output_file
from bokeh.plotting import figure
from bokeh.themes import Theme
from dateutil.parser import parse
from matplotlib import pyplot as plt
from pandas import DateOffset

from Dictionaries import Head_Dict, Scen_Dict


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def odd(n):
    bool = False
    if (n % 2) == 0:
        bool = False
    else:
        bool = True
    return bool


def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def bias(Y_actual, Y_Predicted):
    op = pd.concat([Y_actual, Y_Predicted], axis=1)
    # s.rename()
    op = op.dropna()
    op.columns = ['actual', 'predicted']
    # df=Y_actual.merge(Y_Predicted, left_index=True)
    frac_bias = (op['predicted'] - op['actual']).mean(axis=0) / op['actual'].mean(axis=0)
    return frac_bias


def InitializePlot(obs, pred):
    """
    Initializes the output plot
    with plot formatting options,
    initialized empty datasets,
    etc.
    """
    # split date range
    dr = (pred.index.max() - pred.index.min()) / 2.

    # Create the axes and the figure. Reduce fig size and dpi to make smaller files
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(6.5, 4),
                                   dpi=300,
                                   facecolor="#fcfcfc"
                                   )

    # Set up the axes limits
    ax1.set_xlim(left=pred.index.min(), right=pred.index.min() + dr)
    ax2.set_xlim(left=pred.index.min() + dr, right=pred.index.max())
    # ax.set_ylim(bottom=0, top=15000)

    # Set Axes Formatter
    # ax.xaxis.set_major_formatter(ticker.IndexFormatter(DATE_TICK_INDEX))
    # ax.set_xticks(XTICKS)

    # Set Grid Lines
    ax1.grid(which="both", color="#b5b5b5")
    ax2.grid(which="both", color="#b5b5b5")

    # Set Axes Titles
    ax2.set_xlabel("Date")
    # ax1.set_ylabel("Discharge [cfs]")
    unitText = fig.text(0.01, 0.5, "Discharge [cfs]", va='center', rotation='vertical')
    # Set the year text placeholder
    nseText = ax1.text(
        transform=ax1.transAxes,
        verticalalignment='bottom',
        x=0.01,
        y=.90,
        s="",
        fontsize=8,
        fontfamily='monospace'
    )
    # Set the location text placeholder
    biasText = ax1.text(
        transform=ax1.transAxes,
        verticalalignment='bottom',
        x=0.01,
        y=.83,
        s="",
        fontsize=8,
        fontfamily='monospace'
    )
    # Add Legend
    # ax1.legend(frameon=True, loc='upper right', bbox_to_anchor=(1,1), facecolor='w', fancybox=False, framealpha=1, edgecolor='w')

    return fig, (ax1, ax2), nseText, biasText, unitText


def InitializeMitigationPlot(pred):
    """
    Initializes the output plot
    with plot formatting options,
    initialized empty datasets,
    etc.
    """
    # split date range
    dr = (pred.index.max() - pred.index.min()) / 2.

    # Create the axes and the figure. Reduce fig size and dpi to make smaller files
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(6.5, 8),
                                   dpi=300,
                                   facecolor="#fcfcfc"
                                   )

    # Set up the axes limits
    ax1.set_xlim(left=pred.index.min(), right=pred.index.min() + dr)
    ax2.set_xlim(left=pred.index.min() + dr, right=pred.index.max())
    # ax.set_ylim(bottom=0, top=15000)

    # Set Axes Formatter
    # ax.xaxis.set_major_formatter(ticker.IndexFormatter(DATE_TICK_INDEX))
    # ax.set_xticks(XTICKS)

    # Set Grid Lines
    ax1.grid(which="both", color="#b5b5b5")
    ax2.grid(which="both", color="#b5b5b5")

    # Set Axes Titles
    ax2.set_xlabel("Date")
    # ax1.set_ylabel("Discharge [cfs]")
    yaxisText = fig.text(0.01, 0.5, "Discharge [cfs]", va='center', rotation='vertical')
    # Set the year text placeholder
    nseText = ax1.text(
        transform=ax1.transAxes,
        verticalalignment='bottom',
        x=0.01,
        y=.90,
        s="",
        fontsize=8,
        fontfamily='monospace'
    )
    # Set the location text placeholder
    biasText = ax1.text(
        transform=ax1.transAxes,
        verticalalignment='bottom',
        x=0.01,
        y=.83,
        s="",
        fontsize=8,
        fontfamily='monospace'
    )

    # Add Legend
    ax1.legend(frameon=True, loc='upper right', bbox_to_anchor=(1, 1), facecolor='w', fancybox=False, framealpha=1,
               edgecolor='w')

    return fig, (ax1, ax2), nseText, biasText, yaxisText


def get_file_names(mydir, str=''):
    # gets a list of the xml filenames in the directory for a particular station
    f = []
    # f2=[]
    for root, dirs, files in walk(mydir, topdown=True):
        for x in files:
            if x.endswith(str):
                f.append([os.path.join(root, x), x])
                # f2.append(x)
    return f  # ,f2


def get_scenario_names(mydir, str=''):
    # gets a list of the xml filenames in the directory for a particular station
    scen = []
    path = []
    for root, dirs, files in walk(mydir, topdown=True):
        for x in dirs:
            scen.append(x.split(',')[3])  # the fourth item in the list is the scenario name
            path.append(os.path.join(root, x))
    return scen, path


def Read_RiverWare(fname):
    with open(fname, 'r') as read_file:
        # for i in range(6):
        #     line = read_file.readline().rstrip().split(': ',maxsplit=1)
        #     Head_Dict[line[0]] = line[1]
        line_num = 0
        line = ['nope', 'maybe']
        while line[0][0] != '#':
            line = read_file.readline().rstrip().split(': ', maxsplit=1)
            Head_Dict[line[0]] = line[1]
            line_num += 1
    df = pd.read_csv(fname, skiprows=line_num, header=None)
    # Create index date series from the dates in the RW file dictionary
    dates = pd.date_range(start=Head_Dict['start_date'].split(' ')[0], end=Head_Dict['end_date'].split(' ')[0])
    df.index = dates
    return df, Head_Dict


def Read_RiverSMART(fname):
    with open(fname, 'r') as read_file:
        line = ''
        idx = []
        vals = []
        while line != 'END_RUN_PREAMBLE':  # We don't need any info from the run preamble, so skip for now
            line = read_file.readline().rstrip()
        while line != 'END_SLOT_PREAMBLE':  # get the index series and header info from the slot preamble
            line = read_file.readline().rstrip()
            if is_date(line.split(' ')[0]):  # if it looks like a date, convert and store in index
                idx.append(parse(line.split(' ')[0]))
            else:  # otherwise, it's scenario info.  Read it into a dictionary
                if line != 'END_SLOT_PREAMBLE':
                    line = line.rstrip().split(': ', maxsplit=1)
                    Slot_Dict[line[0]] = line[1]
        for i in range(2):  # Read two more lines into the scenario dictionary
            line = read_file.readline().rstrip().split(': ', maxsplit=1)
            Slot_Dict[line[0]] = line[1]
        ###############################Construct the head dictionary##############################################
        Head_Dict['start_date'] = idx[0].strftime('%Y-%m-%d') + ' 24:00'
        Head_Dict['end_date'] = idx[-1].strftime('%Y-%m-%d') + ' 24:00'
        Head_Dict['units'] = Slot_Dict['units']
        Head_Dict['scale'] = Slot_Dict['scale']
        Head_Dict['# Series Slot'] = Slot_Dict['object_name'] + '.' + Slot_Dict['slot_name'] + ' [' + Slot_Dict[
            'units'] + ']'
        ################################################Read the slot data#############################################
        while str(line).startswith('END') == False:  # get the index series and header info from the slot preamble
            line = read_file.readline().rstrip()
            if is_float(line):
                vals.append(float(line))
    df = pd.DataFrame(vals, index=pd.to_datetime(idx))  # create dataframe from the index and data
    return df, Head_Dict


def ResampleRW(df, Head_Dict, desired_units='acre-feet', desired_scale=1,
               period='A'):  # using the head dictionary and the desired units, resamples to a different timeframe
    cfs2acft = 24 * 60 * 60 / 43560  # conversions from cfs to acre feet
    sf = float(Head_Dict['scale']) / desired_scale  # calculate a scaling factor
    # Find units conversion
    if Head_Dict['units'] == desired_units:
        unit_conversion = 1.
    elif Head_Dict[
        'units'] == 'cfs':  # if we're not using the desired units and we're in cfs, we need to convert to ac-ft
        unit_conversion = cfs2acft  # calculate conversion factor
    elif Head_Dict['units'] == 'acre-feet':
        unit_conversion = 1 / cfs2acft  # calculate conversion factor
    else:
        print('Unit conversion not found for ' + Head_Dict['units'] + ' to ' + desired_units)
    df = df * sf * unit_conversion  # calculate converted dataframe
    df = df.resample(period).sum()
    return df


def GetRWParm(Head_Dict):
    Parm = Head_Dict['# Series Slot'].split('.')[1]  # Get slot and units
    Parm = Parm.split('[')[0]  # drop units
    return Parm


def PlotHeatMap(df, name):
    data = df.copy(deep=False)  # create a copy of the dataframe
    data['Impact'] = -data['Impact']
    ind = data[data.columns[1:]].iloc[0:5, :].mean().sort_values().index
    ind = ind.insert(0, 'Impact')
    data = data.reindex(ind, axis='columns')
    data.to_csv(figpath + name + 'MitigationSorted.csv')
    # data[data.columns[1:]]=data[data.columns[1:]].mean().sort_values().index
    labels = data.columns  # create a copy of the labels
    labels = ['\n'.join(wrap(l, 20)) for l in labels]  # wrap labels for plotting
    data.columns = labels
    plt.figure(figsize=(10, 10))
    heat_map = sns.heatmap(data, linewidth=1, annot=True, fmt='.1f', center=0, cmap="seismic_r",
                           cbar_kws={'label': 'Mitigation, KAF'})
    plt.title(name + " Mitigation")
    plt.subplots_adjust(bottom=0.2, top=0.95)
    # plt.xticks(rotation=45)
    plt.yticks(range(len(data)), data.index.to_list())
    data.to_csv(figpath + name + 'MitigationSorted.csv')
    plt.savefig(figpath + name + 'MitigationHM.png')
    plt.close()


def SetParmLabel(parm):
    label = 'Discharge, [CFS]'
    if parm == "Storage" or parm == 'Pool Elevation':
        label = 'Storage [AF]'
    if parm == 'Pool Elevation':
        label = 'Pool Elevation [Ft]'
    return label


def CreateStyle():
    # Create the style
    style = smgr.create("Plot")
    style.bgColor = 'white'
    style.fgColor = 'black'
    # Figure
    style.figure.width = 10
    style.figure.height = 10
    # Axes
    style.axes.axisBelow = True
    style.axes.leftEdge.color = 'magenta'
    style.axes.leftEdge.width = 5
    util.style = '--'
    style.axes.bottomEdge.color = 'magenta'
    style.axes.bottomEdge.width = 5
    util.style = 'dashed'
    style.axes.topEdge.visible = False
    style.axes.rightEdge.visible = False
    style.axes.title.font.scale = 2.0
    style.axes.title.font.family = 'Segoe UI'
    # X-Axis
    style.axes.xAxis.autoscale = True
    style.axes.xAxis.dataMargin = 0.1
    style.axes.xAxis.label.font.scale = 1.2
    style.axes.xAxis.majorTicks.labels.font.scale = 0.75
    style.axes.xAxis.majorTicks.marks.visible = True
    style.axes.xAxis.majorTicks.grid.visible = True
    style.axes.xAxis.majorTicks.grid.color = '#B0B0B0'
    style.axes.xAxis.majorTicks.grid.width = 1.5
    util.style = ':'
    style.axes.xAxis.majorTicks.length = 15.0
    style.axes.xAxis.majorTicks.width = 1.5
    style.axes.xAxis.minorTicks.marks.visible = True
    style.axes.xAxis.minorTicks.grid.visible = True
    style.axes.xAxis.minorTicks.grid.color = '#B0B0B0'
    style.axes.xAxis.minorTicks.grid.width = 0.5
    util.style = ':'
    style.axes.xAxis.minorTicks.length = 5.0
    style.axes.xAxis.minorTicks.width = 0.5
    # Y-Axis
    style.axes.yAxis = style.axes.xAxis.copy()
    # Lines
    style.line.color = "blue"
    util.style = 'dash-dot'
    style.line.width = 1.5
    style.line.marker.color = 'red'
    style.line.marker.edgeColor = 'green'
    style.line.marker.edgeWidth = 3
    style.line.marker.size = 20
    util.style = 'circle'
    style.line.marker.fill = 'bottom'
    # Patches
    style.patch.color = 'gold'
    style.patch.filled = True
    style.patch.edgeColor = 'purple'
    style.patch.edgeWidth = 5
    # Text
    style.text.lineSpacing = 1.0
    style.text.font.size = 12
    style.text.font.family = 'garamond'
    return style


def initialize_bokeh_plot(df, Head_Dict, Dict, VI_Dict, figpath):
    # Initialize a bokeh plot using a pandas timeseries.  Usually baseline data from scenario
    df.reset_index(inplace=True)
    df.columns = ['index', 'value']
    title = Head_Dict['# Series Slot'].replace('.', ' ')
    title = title.split('[', maxsplit=1)[0]
    p = figure(height=500, width=800, toolbar_location='below',
               outline_line_color='Black',
               min_border_right=10, background_fill_color='White',
               tools='pan,wheel_zoom,box_zoom,reset,save',
               toolbar_sticky=False,
               x_axis_type="datetime", x_axis_location="below", title=
               title
               )
    curdoc().theme = Theme(json=VI_Dict)
    p.line(x='index', y='value', source=df, line_color=Dict['lcolor'], line_dash=Dict['ltype'],
           legend_label=Dict['Name'])
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.yaxis.axis_label = Head_Dict['units']
    output_file(figpath + title + '.html')
    return p

def update_bokeh_plot(p, df, Head_Dict, Dict):
    # Initialize a bokeh plot using a pandas timeseries.  Usually baseline data from scenario
    df.reset_index(inplace=True)
    df.columns = ['index', 'value']
    p.line(x='index', y='value', source=df, line_color=Dict['lcolor'], line_dash=Dict['ltype'],
           legend_label=Dict['Name']) 
    return p


def getHydromet(sta, parm, sd, ed):  # Station code, parameter code, start date, end date
    # function obtains hydromet data from webservice.  req
    # URL = 'https://www.usbr.gov/gp-bin/webarccsv.pl?parameter=' + sta + '%20' + parm + '&syer=' + str(
    #    sd.year) + '&smnth=' + str(sd.month) + '&sdy=' + str(sd.day) + '&eyer=' + str(ed.year) + '&emnth=' + str(
    #    ed.month) + '&edy=' + str(ed.day) + '&format=10'
    # https: // www.usbr.gov / gp - bin / arcread.pl?st = SRLY & by = 1998 & bm = 12 & bd = 31 & ey = 2020 & em = 1 & ed = 3 & pa = QD & json = 1
    URL = 'https://www.usbr.gov/gp-bin/arcread.pl?st=' + sta + '&by=' + str(sd.year) + '&bm=' + str(
        sd.month) + '&bd=' + str(sd.day) + '&ey=' + str(ed.year) + '&em=' + str(ed.month) + '&ed=' + str(
        ed.day) + '&pa=' + parm + '&json=1'
    f = urllib.request.urlopen(URL)
    data = json.load(f)
    d = data['SITE']['DATA']
    df = pd.json_normalize(d)
    df.set_index('DATE', inplace=True)
    df.index = pd.to_datetime(df.index)
    df=df.astype(float)
    return df

def write_trace(df, tracenum, Head_Dict):  # function to write the trace directory files
    directory = outdir + df.index[0].strftime("%Y%m%d") + '\\trace' + str(tracenum)
    fname = directory + '\\Buffalo_Bill_Inflows.local.rdf'
    # write the trace directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(fname, 'w') as fout:
        # writer = csv.writer(fout)
        # for k, v in Head_Dict.items():
        #     writer.writerow(v)
        for key in Head_Dict.keys():
            fout.write("%s\n" % (Head_Dict[key]))
        if (isinstance(tracenum, int)):
            for index, row in df.iterrows():
                fout.write("%f\n" % (row[tracenum - 1]))
        else:
            for index, row in df.iterrows():
                fout.write("%f\n" % (row[tracenum]))
        # df[column].to_csv(fname,index=None,header=False,mode='a+')


def getHDB(svr='ecohdb', sdi='100840', tstp='DY', sd=pd.to_datetime("today") - DateOffset(months=1),
           ed=pd.to_datetime("today"),
           format='json'):  # Station code, parameter code, start date, end date
    # function obtains hydromet data from webservice.  req
    # svr -lchdb, Lower Colorado Regional Office;
    # uchdb2, Upper Colorado Regional Office
    # yaohdb,  Yuma Area Office
    # ecohdb, Eastern Colorado Area Office
    # lbohdb, Lahontan Basin Area Office
    # kbohdb, Klamath Basin Area Office
    # pnhyd, Pacific Northwest Regional Office
    # or gphyd Great Plains Regional Office Hydromets
    # sdi: Site datatype ID
    # tstp - IN, HR, DY, MN to query instantaneous, hourly, daily, or monthly data
    # sd: start datetime
    # ed: end datetime
    # format - csv, table, json, or graph

    # Format sd and ed as strings
    if tstp == 'DY' or tstp == 'MN':
        t1 = sd.strftime('%m-%d-%Y')
        t2 = ed.strftime('%m-%d-%Y')
    else:
        t1 = sd.tz_convert('America/Denver').strftime('%m-%d-%YT%H:%M')
        t2 = ed.tz_convert('America/Denver').strftime('%m-%d-%YT%H:%M')
    # format the url string
    URL = 'https://www.usbr.gov/pn-bin/hdb/hdb.pl?svr=' + svr + '&sdi=' + sdi + '&tstp=' + tstp + '&t1=' + t1 + '&t2=' + t2 + '&table=R&mrid=0&format=json'
    # retrieve file

    if format == 'json':
        f = urllib.request.urlopen(URL)
        # read json file
        data = json.load(f)
        # extract timeseries data. Currently set up to read one timeseries.
        d = data['Series'][0]['Data']
        df = pd.json_normalize(d)
        df['v'] = pd.to_numeric(df['v'])
        df.set_index('t', inplace=True)
        df.index = pd.to_datetime(df.index)
        colname = [data['Series'][0]['SiteName'] + ' ' + data['Series'][0]['DataTypeName']]
        df.columns = colname
    if format == 'csv':
        df = pd.read_csv(URL)  # ,skiprows=1)#,header=14,index_col=0,parse_dates=True)
        # df = pd.read_json(URL)
        # print(data)
    return df
