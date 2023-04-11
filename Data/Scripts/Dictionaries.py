cfs2acft = 24 * 60 * 60 / 43560  # conversions from cfs to acre feet
Head_Dict = {'start_date': '1980-01-01 24:00',
             'end_date': '2021-06-30 24:00',
             # 'data_date': '1980-01-01 24:00',
             'timestep': '1 DAY',
             'units': 'cfs',
             'scale': '1.000000',
             '# Series Slot': 'Buffalo Bill Inflows.Local Inflow [1 cfs]'}
Slot_Dict = {'object_type': 'StreamGage',
             'object_name': 'Battle Creek at International Boundary',
             'slot_name': 'Gage Outflow',
             'units': 'cfs',
             'scale': '1'}

#Dictionary of the scenarios to be analyzed, and plotting parameters.
Scen_Dict = {'Baseline': {'Folder': 'Baseline', 'Name': 'Baseline', 'lcolor': 'Black', 'ltype': 'solid', 'Plot': False},
             'Balanced': {'Folder': 'Balanced', 'Name': 'Balanced Operations', 'lcolor': 'Black',
                          'ltype': 'solid', 'Plot': True},
             'FloodControl': {'Folder': 'FloodControl', 'Name': 'Minimize Flooding', 'lcolor': '#FF671F', 'ltype': 'solid',
                              'Plot': True},
             'Hydropower': {'Folder': 'Hydropower', 'Name': 'Emphasize Hydropower Generation', 'lcolor': '#215732',
                            'ltype': 'dashed', 'Plot': True},
             'PyForecast': {'Folder': 'PyForecast', 'Name': 'Seasonal Statistical Forecasts', 'lcolor': '#4C12A1',
                            'ltype': 'dashdot', 'Plot': False},
             'Historical': {'Folder': 'Historical', 'Name': 'Historical Traces', 'lcolor': '#9A3324', 'ltype': 'dashdot',
                            'Plot': False},
             }
# VI Colors: Dark Blue,'#003E51'
#             Light Blue,'#007396'
#             Mustard,'#C69214'
#             Tan,'#DDCBA4'
#             Orange,'#FF671F'
#             Green,'#215732'
#             Purple,'#4C12A1'
#             Red,'#9A3324'