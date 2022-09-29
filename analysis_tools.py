import query_data_from_DB as bd
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pretty_html_table import build_table
import warnings
import traceback
from scipy.interpolate import CubicSpline
from scipy.stats import linregress
import scipy.signal



pd.options.mode.chained_assignment = None




class SQC_tools():


 def __init__():
 
   pass



 def make_list(self, filename):

    with open(filename) as f:

        list_with_values = json.load(f)

    split_data = []
    
    for j in list_with_values:
        split_data.append(j.split(','))

    return split_data




 def make_dataframe(self, file_prefix, sql_parameter, headers):

   filename = '{}_data/{}.json'.format(file_prefix, sql_parameter)

   datalist = self.make_list(filename)
   df = pd.DataFrame(datalist, columns=headers)
   if sql_parameter !='runs' and file_prefix=='SQC':
       # drop data from older sensors
       
       df = df.loc[(~df['Sensor'].str.contains('HPK'))]
       df = df.loc[~df['Sensor'].str.contains('33234')]

   return df



 def filter_dataframe(self, df, parameter, lower_limit, upper_limit):

   df[parameter] = pd.to_numeric(df[parameter]).abs()

   df = df.loc[(df[parameter]<float(upper_limit)) & (df[parameter]>float(lower_limit))]

   return df




 def find_wafer_type(self, df, header):

    column_name = '{}_type'.format(header)

    df[column_name] = np.nan

    df[column_name] = df[column_name].mask(df[header].str.contains('2-S'), '2-S').mask(df[header].str.contains('PSS'), 'PSS').mask(df[header].str.contains('PSP'), 'PSP')

    return df



 def find_batch(self, df, header):

    df['Batch'] = np.nan

    df['Batch'] = (df[header].str.split('_')).str[0]
    df['Batch'] = pd.to_numeric(df['Batch'])

    df = df.sort_values(by=['Batch'])

    df['Batch'] = df['Batch'].astype(str)

    return df




 def find_location_SQC(self, df_location, df_data, header):

   
    merged_df = pd.merge(df_data, df_location, how='outer', on=['Run_number'])

    merged_df = merged_df.dropna(subset=[header, 'Run_number'])
    

    return merged_df





 def merge_dataframes(self, parameter_df, metadata_df, volts, parameter):


    parameter_df['Condition_number'] = pd.to_numeric(parameter_df['Condition_number'])

    metadata_df['Condition_number'] = pd.to_numeric(metadata_df['Condition_number'])

    parameter_df = parameter_df.sort_values(by=['Condition_number'])
    metadata_df = metadata_df.sort_values(by = ['Condition_number'])
    

    
    parameter_df['Condition_number'] = parameter_df['Condition_number']- 1


 
    merged_df = pd.merge(metadata_df, parameter_df[['Condition_number', str(volts), str(parameter)]], on='Condition_number', how='left') 

    merged_df = merged_df.dropna(subset=[parameter])
    

    return merged_df






 def sqc_sequence(self, df, runs_df):


    df = self.find_wafer_type(df, 'Sensor')
    df = self.find_batch(df, 'Sensor')
    df = self.find_location_SQC(runs_df, df, 'Sensor')

    return df


 def convert_serial_from_binary(self, df):
    
       #from bitstring import BitArray
      
       df = df.loc[(df['Serial_number'].str.contains('1')) & (~df['Serial_number'].str.contains('Test'))]
       serial_numbers = df['Serial_number'].values
       
       
       
       def debinary(i):
         #for i in serial_numbers:
           binary = i.split('|')[1:7]
        
           try:
               binary = [b.replace(b, str(int(b,2))) for b in binary]
           
           except ValueError:           
               binary = i.split('|')[1:6]
               binary = [b.replace(b, str(int(b,2))) for b in binary]
           
           return ''.join(binary)[2:]  
        
       df['Serial_number'] = df['Serial_number'].apply(lambda x: debinary(x))
       
       return df
    
    
    
    
 def find_sensor(self, df):
    
      df['Sensor_number'] = np.nan
      df['Sensor_number'] = df['Sensor'].apply(lambda x: '_'.join(x.split('_')[1:2]))
    
      return df


 def make_html_table(self, df, color, name):


    html_table = build_table(df, color, text_align='center')
    with open('html_tables/{}.html'.format(name), 'w') as f:
       # f.write('<html><body><h1> {} outside of specifications <font color = #4000FF></font></h1>\n</body></html>')
       # f.write('\n')
        f.write(html_table)




 def find_SQC_values_out_of_specs(self, df, parameter, config_file):

   
    list_with_condition_larger_than = ['Cac', 'Rint']

    if parameter in list_with_condition_larger_than:

        bad_strips_df = df.loc[df[parameter]<config_file['SQC_parameters'][parameter]['Limit']]

    elif parameter== 'Rpoly':

        bad_strips_df = df.loc[~df[parameter].between(config_file['SQC_parameters'][parameter]['Limit']-0.5, config_file['SQC_parameters'][parameter]['Limit']+0.7)]

    else:

        bad_strips_df = df.loc[df[parameter].abs()>config_file['SQC_parameters'][parameter]['Limit']]

    return bad_strips_df




 def find_outliers(self, df, config_file, parameter):


    upper_limit = config_file['up']
    lower_limit = config_file['low']


    return upper_limit, lower_limit



 def analyse_cv(self, v, c, area=1.56e-4, carrier='electrons', cut_param=0.008, max_v=500, savgol_windowsize=None, min_correl=0.1, debug=False):
    """
    Diode CV: Extract depletion voltage and resistivity.
    Parameters:
    v ... voltage
    c ... capacitance
    area ... implant size in [m^2] - defaults to quarter
    carrier ... majority charge carriers ['holes', 'electrons']
    cut_param ... used to cut on 1st derivative to id voltage regions
    max_v ... for definition of fit region, only consider voltages < max_v
    savgol_windowsize ... number of points to calculate the derivative, needs to be odd
    min_correl ... minimum correlation coefficient to say that it worked
    Returns:
    v_dep1 ... full depletion voltage via inflection
    v_dep2 ... full depletion voltage via intersection
    rho ... resistivity
    conc ... bulk doping concentration
    """

    # init
    v_dep1 = v_dep2 = rho = conc = np.nan
    a_rise = b_rise = a_const = b_const = np.nan
    v_rise = []
    v_const = []
    status = 'Pass'

    if savgol_windowsize is None:
        # savgol_windowsize = int(len(c) / 40 + 1) * 2 + 1  # a suitable off windowsie - making 20 windows along the whole measurement
        savgol_windowsize = int(len(c) / 30 + 1) * 2 + 1  # a suitable off windowsie - making 15 windows along the whole measurement
        # the window size needs to be an odd number, therefore this strange calculation

    # invert and square
    #c = [1./i**2 for i in c]

    # get spline fit, requires strictlty increasing array
    y_norm = c / np.max(c)
    x_norm = np.arange(len(y_norm))
   
    spl_dev = scipy.signal.savgol_filter(y_norm, window_length=savgol_windowsize, polyorder=1, deriv=1)

    # for definition of fit region, only consider voltages < max_v
    idv_max = max([i for i, a in enumerate(v) if abs(a) < max_v])
    spl_dev = spl_dev[:idv_max]

    idx_rise = []
    idx_const = []

    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        try:
            # get regions for indexing
            idx_rise = [i for i in range(2, len(spl_dev-1)) if ((spl_dev[i]) > cut_param)]  # the first and last value seems to be off sometimes
            idx_const = [i for i in range(2, len(spl_dev-1)) if ((spl_dev[i]) < cut_param) and i > idx_rise[-1]]

            v_rise = v[idx_rise[0]:idx_rise[-1] + 1]
            v_const = v[idx_const[0]:idx_const[-1] + 1]
            c_rise = c[idx_rise[0]:idx_rise[-1] + 1]
            c_const = c[idx_const[0]:idx_const[-1] + 1]

            # line fits to each region
            a_rise, b_rise, r_value_rise, p_value_rise, std_err_rise = scipy.stats.linregress(v_rise, c_rise)
            a_const, b_const, r_value_const, p_value_const, std_err_const = scipy.stats.linregress(v_const, c_const)

        
            mu = 1350*1e-4 

            # full depletion voltage via max. 1st derivative
            v_dep1 = v[np.argmax(spl_dev)]

            # full depletion via intersection
            v_dep2 = (b_const - b_rise) / (a_rise - a_const)
            
            conc = 2. / (1.6e-19 * 11.68 *8.854e-12 * a_rise * area**2) 
            rho = 1. / (mu*1.6e-19 *conc)



        except np.RankWarning:
            
            print("The array has too few data points. Try changing the cut_param parameter.")

        except (ValueError, TypeError, IndexError):
        
            print("The array seems empty. Try changing the cut_param parameter.")

        if status == 'Fail':
            #print("The fit didn't work as expected, returning nan")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, STATUS_FAILED
    
    return v_dep1, v_dep2, rho





 def plot_distribution(self, df, config_file, parameter, plot_type):

    from matplotlib.patches import Rectangle

    df[parameter]= df[parameter].abs()  
    low, upper = find_outliers(df, config_file, parameter)

    fig, ax = plt.subplots()
    

    hist, bins, patches = plt.hist(df[parameter], bins=80, range=(low, upper), color='blue', edgecolor='black', linewidth=1.2)
    patches[0].set_height(patches[0].get_height() + df.loc[df[parameter]<=low].shape[0])
    patches[0].set_facecolor('green')
    patches[-1].set_height(patches[-1].get_height() + df.loc[df[parameter]>=upper].shape[0])
    patches[-1].set_facecolor('yellow')

    ax.set_xlabe('{}'.format(config_file), fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of {}'.format(y_label), fontsize=12, fontweight='bold')
    extra = Rectange((0,0), 1, 1, fc='q', fill=False, edgecolor='none', linewidth=0)
    #label = 'Specs:

   # leg = plt.legend([extra], [label], frameon=False, loc='best')

    plt.savefig('{}_{}.png'.format(parameter, plot_type))




 def plot_time_evolution(self, df, parameter, measured_entity, ylabel):


    batch = list(dict.fromkeys(df['Batch']))


    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Batch', y=parameter, hue=measured_entity, s=25, linewidth=0, ax=ax)
    plt.xticks(batch, batch, rotation=90, va='top', fontsize=6)
    plt.locator_params(axis='x', nbins=len(batch)/4)
    ax.set(xlabel=None)
    if parameter == 'vdp_bulk' or parameter=='Diode_bulk':
        plt.ylim(0,11)
        plt.axhline(y=3.5, linestyle='dashed', color='black')
   #plt.axvline(x=16, linestyle='dashed', color='black')
    plt.axhline(y=df[parameter].median(), linestyle='dashed', color='black')

    #plt.ylim(100, 400) 
    plt.ylabel(ylabel)
    plt.legend(loc='best')
   
    plt.savefig('{}_evolution.png'.format(parameter))
 
 
 
 
 def plot_time_evolution_as_bookeh(self, df, parameter, measured_entity, ylabel):

    from bokeh.plotting import figure, show
    from bokeh.models import SingleIntervalTicker, LinearAxis, FixedTicker
    import holoviews as hv
    import hvplot
    import hvplot.pandas
    import holoviews.plotting.bokeh
    from holoviews.operation.timeseries import rolling, rolling_outlier_std



# If you are using bokeh as a backend you can also just use 'color' parameter.
# I like this one more because it creates a hv.Scatter() instead of hv.NdOverlay() 
# 'category_col' is here just an extra vdim, which is used for colors
    hv.extension('bokeh')
    #pd.options.plotting.backend = 'hvplot'
    batch = list(dict.fromkeys(df['Batch']))
    print(df.to_string())
    df['Batch'] = pd.to_numeric(df['Batch'])
    #p = figure()#plot_width=800, plot_height=600)#x_range=batch, y_range=(0, 1000))
   
    #p.xaxis.major_label_orientation = np.pi/2   # radians, "horizontal", "vertical", "normal"
    #p.xaxis.ticker = SingleIntervalTicker(interval=len(b)/2)
    #p.xaxis.visible = False

    # Add custom axis
   
    #p.star(x=df['Sensor'], y= df[parameter], color = "#ff1200")
    #graph= hv.Scatter(data=df, kdims=['Batch'], vdims=[parameter, 'Sensor_type']).opts(color='Sensor_type', cmap=['blue', 'orange', 'green'])
    #fig, ax = plt.subplots()
   
    
    #graph= sns.scatterplot(data=df, x='Batch', y=parameter, hue=measured_entity, s=25, linewidth=0)
    
    
    
    #graph = df.hvplot(x='Batch', y=parameter, kind='scatter', height=600, width = 800)
    #def load_symbol(symbol, variable, **kwargs):
    #     graph = df.hvplot(x='Sensor', y=parameter, kind='scatter', by='Sensor_type', height=600, width = 800)
         
     #    return  graph #hv.Curve(df, ('date', 'Date'), variable).opts(framewise=True)

  
    
    
    
   # dmap = hv.DynamicMap(load_symbol, kdims=['Type','Batch'])
   # dmap = dmap.redim.values(Type=df['Sensor_type'], Batch= df['Sensor'])

   # dmap.opts(framewise=True)
    #sa= rolling(dmap, rolling_window=2)
    
    #graph = figure(title = "CMS Tracker data") 
    #graph.scatter(x=df['Batch'], y=df[parameter], color='red')   
    #my_bokeh_plot = hv.render(graph, backend='bokeh')
    show(hv.render(dmap))
      

   
   

 

class PQC_tools:


  def __init__():
  
     pass


  def make_list_with_data(self, json_filename):

     
     with open('PQC_data/{}.json'.format(json_filename)) as f:

         lista = json.load(f)

     splt = []

     for i in lista:

        splt.append(i.split(','))

  
     return splt




  def make_dataframe(self, json_filename, headers_list):

    print(json_filename)
    data_list = self.make_list_with_data(json_filename)

    df = pd.DataFrame(data_list, columns=headers_list)

    return df



  def find_location_PQC(self, df):

   
    df['Location'] = np.nan
    df['Location'] = df['Location'].mask(df['File'].str.contains('HPK_'), 'Perugia')

    df['Location'] = df['Location'].mask(df['File'].str.contains('Test'), 'Brown').mask(df['File'].str.contains('Demokritos'), 'Demokritos').mask(df['File'].str.contains('.json'), 'HEPHY')

    return df
    

  def filter_data(self, df, parameter):

    df[parameter] = pd.to_numeric(df[parameter]).abs()
    df = df.dropna(subset=[parameter])
    
    return df




  def find_batch(self, df):


    df['Batch'] = np.nan
    df['Batch'] = df['Halfmoon'].str.split('_').str[0]

    df['Batch'] = pd.to_numeric(df['Batch'])
    df = df.sort_values(by=['Batch'])
    df['Batch'] = df['Batch'].astype(str)

    return df




  def find_wafer_type(self, df):

     df['HM_type'] = np.nan
     df['HM_type'] = df['HM_type'].mask(df['Halfmoon'].str.contains('2-S'), '2-S').mask(df['Halfmoon'].str.contains('PSS'), 'PS-S').mask(df['Halfmoon'].str.contains('PSP'), 'PS-P')


     return df



  def merge_PQC_dataframes(self, parameter_df, metadata_df, parameter): #volts


    parameter_df['Condition_number'] = pd.to_numeric(parameter_df['Condition_number'])

    metadata_df['Condition_number'] = pd.to_numeric(metadata_df['Condition_number'])

    parameter_df = parameter_df.sort_values(by=['Condition_number'])
    metadata_df = metadata_df.sort_values(by = ['Condition_number'])
    

    
    parameter_df['Condition_number'] = parameter_df['Condition_number']- 2


   
    merged_df = pd.merge(metadata_df, parameter_df[['Condition_number', str(parameter)]], on='Condition_number', how='left') #str(volts)

    merged_df = merged_df.dropna(subset=[parameter])
    

    return merged_df



  def pqc_sequence(self, df, metadata_df, parameter):

    list_of_attributes = [self.find_batch, self.find_wafer_type]

     
    df = self.merge_PQC_dataframes(df, metadata_df, parameter)

    
    df = self.find_location_PQC(df)
    
    for attr in list_of_attributes:
          df = attr(df)

    
    df[parameter] = pd.to_numeric(df[parameter])
    
    df[parameter] = df[parameter].abs()
    df = df.dropna(subset=[parameter])


    return df
    
    
    
  def find_interquartile(self, df, parameter):

       median = df[parameter].median()
       q3, q1 = np.percentile(df[parameter], [75,25])
       inter_q = q3 - q1
       
        #self.find_MAD(final_df, parameter))

       return q1,q3

 

  def make_html_table(self, df, parameter):

      html_table = build_table(df, 'blue_light', text_align='center')
      with open('{}_halfmoons_outliers.html'.format(parameter), 'w') as f:
          f.write('<html><body> <h1>Halfmoons with measured outlier values <font color = #4000FF></font></h1>\n</body></html>')
          f.write('\n')
          f.write(html_table)



  def analyse_cv(self, v, c, area=1.56e-4, carrier='electrons', cut_param=0.008, max_v=500, savgol_windowsize=None, min_correl=0.1, debug=False):
    """
    Diode CV: Extract depletion voltage and resistivity.
    Parameters:
    v ... voltage
    c ... capacitance
    area ... implant size in [m^2] - defaults to quarter
    carrier ... majority charge carriers ['holes', 'electrons']
    cut_param ... used to cut on 1st derivative to id voltage regions
    max_v ... for definition of fit region, only consider voltages < max_v
    savgol_windowsize ... number of points to calculate the derivative, needs to be odd
    min_correl ... minimum correlation coefficient to say that it worked
    Returns:
    v_dep1 ... full depletion voltage via inflection
    v_dep2 ... full depletion voltage via intersection
    rho ... resistivity
    conc ... bulk doping concentration
    """

    # init
    v_dep1 = v_dep2 = rho = conc = np.nan
    a_rise = b_rise = a_const = b_const = np.nan
    v_rise = []
    v_const = []
    status = 'Pass'

    if savgol_windowsize is None:
        # savgol_windowsize = int(len(c) / 40 + 1) * 2 + 1  # a suitable off windowsie - making 20 windows along the whole measurement
        savgol_windowsize = int(len(c) / 30 + 1) * 2 + 1  # a suitable off windowsie - making 15 windows along the whole measurement
        # the window size needs to be an odd number, therefore this strange calculation

    # invert and square
    #c = [1./i**2 for i in c]

    # get spline fit, requires strictlty increasing array
    y_norm = c / np.max(c)
    x_norm = np.arange(len(y_norm))
   
    spl_dev = scipy.signal.savgol_filter(y_norm, window_length=savgol_windowsize, polyorder=1, deriv=1)

    # for definition of fit region, only consider voltages < max_v
    idv_max = max([i for i, a in enumerate(v) if abs(a) < max_v])
    spl_dev = spl_dev[:idv_max]

    idx_rise = []
    idx_const = []

    with warnings.catch_warnings():
        warnings.filterwarnings('error')

        try:
            # get regions for indexing
            idx_rise = [i for i in range(2, len(spl_dev-1)) if ((spl_dev[i]) > cut_param)]  # the first and last value seems to be off sometimes
            idx_const = [i for i in range(2, len(spl_dev-1)) if ((spl_dev[i]) < cut_param) and i > idx_rise[-1]]

            v_rise = v[idx_rise[0]:idx_rise[-1] + 1]
            v_const = v[idx_const[0]:idx_const[-1] + 1]
            c_rise = c[idx_rise[0]:idx_rise[-1] + 1]
            c_const = c[idx_const[0]:idx_const[-1] + 1]

            # line fits to each region
            a_rise, b_rise, r_value_rise, p_value_rise, std_err_rise = scipy.stats.linregress(v_rise, c_rise)
            a_const, b_const, r_value_const, p_value_const, std_err_const = scipy.stats.linregress(v_const, c_const)

        
            mu = 1350*1e-4 

            # full depletion voltage via max. 1st derivative
            v_dep1 = v[np.argmax(spl_dev)]

            # full depletion via intersection
            v_dep2 = (b_const - b_rise) / (a_rise - a_const)
            
            conc = 2. / (1.6e-19 * 11.68 *8.854e-12 * a_rise * area**2) 
            rho = 1. / (mu*1.6e-19 *conc)



        except np.RankWarning:
            
            print("The array has too few data points. Try changing the cut_param parameter.")

        except (ValueError, TypeError, IndexError):
        
            print("The array seems empty. Try changing the cut_param parameter.")

        if status == 'Fail':
            #print("The fit didn't work as expected, returning nan")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, STATUS_FAILED
    
    return v_dep1, v_dep2, conc



  def histogram(self, df, parameter, to_draw):

     if parameter=='vdp_bulk':
         df= df.loc[~df['Location'].str.contains('Perugia')]

     MAD = ((df[parameter] - df[parameter].median()).abs()).median()
     low = self.config_file['PQC_parameters'][str(to_draw)]['lower']
     upper =  self.config_file['PQC_parameters'][str(to_draw)]['upper']
     
     batch = list(dict.fromkeys(df['Batch']))
   
     
     print(df.loc[df[parameter]<=low])
     print(df.loc[df[parameter]<=low].shape[0])   

     fig, ax = plt.subplots()

 
     hist, bins, patches = plt.hist(df[parameter].abs(), bins='auto', range=(low, upper), color='red', edgecolor='black', linewidth=1.2)
     
     print(patches[0].get_height())  
     patches[0].set_height(df.loc[df[parameter]<=low].shape[0])
     patches[0].set_facecolor('blue')
     patches[-1].set_height(df.loc[df[parameter]>=upper].shape[0])
     patches[-1].set_facecolor('yellow')

     ax.set_xlabel('{} [{}]'.format(self.config_file['PQC_parameters'][str(to_draw)]['ylabel'], self.config_file['PQC_parameters'][str(to_draw)]['units']), fontsize=12, fontweight = 'bold')

     ax.set_ylabel('Number of test structures', fontsize=12, fontweight='bold')        

     plt.savefig('Figures/{}_histogram.png'.format(to_draw))

     plt.clf()



  def plot_time_evolution(self, df, parameter):

     from matplotlib.patches import Rectangle
  
     
     MAD = ((df[parameter] - df[parameter].median()).abs()).median()
     
      
     low = self.config_file['PQC_parameters'][str(parameter)]['lower']
     upper =  self.config_file['PQC_parameters'][str(parameter)]['upper']
     
     df_out_of_range = df.loc[(df[parameter]>upper) | (df[parameter]<low)]
     
     
     def plot_histo_with_outliers(df):     
         
         plt.clf()
     
         sns.histplot(data= df_out_of_range, x= 'Batch', hue='Location')
         plt.xticks(rotation=90, va='top', fontsize = 6)
 
         plt.ylabel('Number of halfmoons')
         plt.title('{} - out of typical values'.format(parameter))
         plt.savefig('{}_histogram_out_of_specs.png'.format(parameter))
         plt.clf()
     
     
     plot_histo_with_outliers(df_out_of_range)
     
     df = df.loc[(df[parameter]<=upper) & (df[parameter]>=low)]
     
     batch = list(dict.fromkeys(df['Batch']))     
     fig, ax = plt.subplots()
      
     sns.scatterplot(data=df, x='Batch', y=parameter, hue = 'HM_type', s=25, linewidth=0, ax=ax) #, hue='Location'
     plt.xticks(batch, batch, rotation=90, va = 'top', fontsize=6)
     plt.locator_params(axis='x', nbins = len(batch)/4)
     ax.set_title('CMS Tracker data', fontsize= 15, fontweight = 'bold')
     ax.set(xlabel=None)
     
     ax.set_ylabel('{} [{}]'.format(self.config_file['PQC_parameters'][str(parameter)]['ylabel'], self.config_file['PQC_parameters'][str(parameter)]['units']), fontsize=12, fontweight = 'bold')
 
     #plt.ylim(low, upper)
     #ax2 = plt.axes([0.3, 0.5, .7, .3], facecolor='c')
     #ax2.set_xlabel('None')
     #sns.scatterplot(data=df, x='Batch', y=parameter, hue='Location', s=25, linewidth=0, ax=ax2)
     #ax2.set_ylim([0.1*df[parameter].median(), 2*df[parameter].median()])
     #label = 'Median: {} {}\nstdev: {} {}'.format(np.round(df[parameter].median(),2), self.config_file['PQC_parameters'][str(parameter)]['units'], np.round(df[parameter].std(),2), self.config_file['PQC_parameters'][str(parameter)]['units'])
     #ax.annotate(label, xy=(0.95, 0.9), xycoords='axes fraction', fontsize=11,
             #   horizontalalignment='right', verticalalignment='bottom', color='red')
 
     plt.savefig('{}_filtered_location.pdf'.format(parameter))
