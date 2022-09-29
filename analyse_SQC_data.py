import subprocess
import query_data_from_DB as db
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pretty_html_table import build_table
import warnings
from scipy.interpolate import CubicSpline
from analysis_tools import SQC_tools

#from analyse_PQC_data import vdp_bulk

pd.options.mode.chained_assignment = None


########################################################################################################################################


class runs(SQC_tools):

# Class which process the SQC information fetced from runs SQL table. It includes the run_number 
# and most importantly the location of the QC center


    def __init__(self):

        self.measurement = 'SQC'



    def runs_dataframe(self):

      runs_df = SQC_tools.make_dataframe(self, self.measurement, 'runs', ['Time', 'Location', 'Type', 'Run_number'])
      
      # Filter out the PQC data
      runs_df = runs_df.loc[runs_df['Type'].isin(['SQC', 'VQC'])].drop_duplicates(subset=['Run_number', 'Location'], keep='first')
     
   
      return runs_df


#########################################################################################################################################




class bad_Strips(SQC_tools):

 # This class analyses the data which correspond to the bad strips detected by HPK



    def __init__(self):

        self.measurement = 'SQC'



    def run(self):

      bad_strips_df = SQC_tools.make_dataframe(self, self.measurement, 'bad_strips', ['Sensor', 'Defect', 'Strip'])

       

      return bad_strips_df



###########################################################################################################################################



class SQC_summary(SQC_tools):

    # This class produces html tables with the information of sensors with known problems or bad sensors, tested at SQC centers 

    
    def __init__(self, df_sqc_data):

        self.measurement = 'SQC'
        self.df_sqc_data = df_sqc_data.loc[df_sqc_data['Type']=='SQC']



    def find_sensors_with_problems(self, df, df_sqc_data, sensor_type):

        df = df.loc[df['Known_problem']!='']
        
        df = pd.merge(df, df_sqc_data[['Sensor', 'Location']], how='left', on=['Sensor'])
                
        SQC_tools.make_html_table(self, df, 'red_dark', '{}_sensors_with_problems'.format(sensor_type))



    def find_bad_sensors(self, df, df_sqc_data, sensor_type):


        df = df.loc[df['Status']=='Bad']


        df = pd.merge(df, df_sqc_data[['Sensor', 'Location']], how='left', on=['Sensor'])
        
        SQC_tools.make_html_table(self, df, 'red_dark', '{}_bad_sensors'.format(self.measurement, sensor_type))
      
      
    
       

    def run(self):

       dataframe_list = []
       
       for sensor_type in ['2-S', 'PS-s']: 
           
           summary_df = SQC_tools.make_dataframe(self, self.measurement, 'summary_{}_sensors'.format(sensor_type), ['Sensor', 'Batch', 'Serial_number', 'Description', 'Sensor_type', 'Status', 'Known_problem'])
           summary_df = SQC_tools.find_sensor(self, summary_df)
           summary_df = SQC_tools.convert_serial_from_binary(self, summary_df)
           
           
           
           self.find_bad_sensors(summary_df, self.df_sqc_data, sensor_type)
           

           self.find_sensors_with_problems(summary_df, self.df_sqc_data, sensor_type)

           dataframe_list.append(summary_df)

       return dataframe_list  


##################################################################################################################
##################################################################################################################
#########################                 IV                ######################################################
##################################################################################################################
##################################################################################################################




class IV(runs):



    def __init__(self, iv):

       
        self.iv = iv
        
        self.config_file = db.read_yml_configuration('SQC_parameters_DB.yml')
        self.measurement = 'SQC'

        


    def scale_current(self, df):
       
        ##scales the current at 21oC

        df[[self.iv, 'Temp']] = df[[self.iv, 'Temp']].apply(pd.to_numeric, errors='coerce').abs()

        df['Temp']  = df['Temp'].mask(df['Temp'].isnull()==True, float(25)) # assigns a temperature of 25oC to HPK data
    
        df['I_scaled'] = (df[self.iv]*21)/df['Temp']

        return df




    def make_table_with_outliers(self, df):


        outliers_df = df.loc[df['I_scaled']>self.config_file['SQC_parameters']['IV']['Limit']]

        tools.make_html_table(outliers_df)



    def plot_batches_over_time(self, df):



        df_HPK = df.loc[df['Location']=='Hamamatsu']

        df_HPK = df_HPK.groupby(['Batch', 'Sensor_type']).agg(Wafers=('Batch', 'count'))
        df_HPK = df_HPK.reset_index()
      
        df_HPK['Wafers'] = pd.to_numeric(df_HPK['Wafers'])

        df_2S_only = df_HPK.loc[df_HPK['Sensor_type'].str.contains('2-S')]

        df_PSS_only = df_HPK.loc[df_HPK['Sensor_type'].str.contains('PSS')]


        low_yield_2S_df = df_2S_only.loc[df_2S_only['Wafers']<30]

        low_yield_PSS_df = df_PSS_only.loc[df_PSS_only['Wafers']<75]

        fig, ax = plt.subplots()
        sns.scatterplot(data=df_HPK, x='Batch', y='Wafers', hue='Sensor_type', s=27, linewidth=0, ax=ax)
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::len(list(dict.fromkeys(df['Batch'])))//40])  
        ax.tick_params(axis='x', rotation=90, labelsize = 6)
        ax.set(xlabel=None)
        plt.axhline(y=75, linestyle='dashed', color='blue') 
        plt.axhline(y=30, linestyle='dashed', color='orange')
        plt.text(26, 65, 'PS-s low yield batches: {}/{}'.format(low_yield_PSS_df.shape[0], df_PSS_only.shape[0]), color='blue', style='italic', fontsize=11)

        plt.text(26, 25, '2-S low yield batches: {}/{}'.format(low_yield_2S_df.shape[0], df_2S_only.shape[0]), color='orange', style='italic', fontsize=11)
        ax.set_ylabel('Number of Sensors', fontsize=12, fontweight='bold')
        plt.legend(loc='best')
        plt.savefig('Figures/{}/batches_over_time.pdf'.format(self.measurement))

             


    def plot_itot_over_time(self, df, label):


        ## plots the scaled current at -600(-800) V over batch number
  
        batch = list(dict.fromkeys(df['Batch']))
       
        df = df.loc[df['I_scaled']>10]
        df['Time'] = pd.to_datetime(df['Time'], errors = 'coerce')
        
        #df['New'] = df.groupby('Sensor')['I_scaled']#.diff()
              
        df = df.sort_values('Time').drop_duplicates(['Sensor', 'Location'], keep='last')
        
   
        
        fig, ax = plt.subplots()
        sns.scatterplot(data = df, x='Batch', y='I_scaled', hue='Sensor_type', s=25, linewidth=0, ax=ax)
        
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::len(batch)//40])  
        ax.tick_params(axis='x', rotation=90, labelsize = 6)
        
        ax.set(xlabel=None)

        plt.axvline(x=22, linestyle='dashed', color='black')
        plt.axhline(y = 3125, linestyle ='dashed', color='blue')
        
        plt.axhline(y = 7250, linestyle ='dashed', color='orange')
        plt.text(26, 2000, 'PS-s/PS-p limit', color = 'blue', style ='italic', fontsize = 12)
       
        plt.text(26, 8500, '2-S limit', color = 'orange', style ='italic', fontsize = 12)
        plt.text(26, 1000, 'Production Period', style ='italic')
        plt.yscale('log')

        #plt.figtext(0.4, 0.01, r'Time evolution $\longrightarrow$', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Current@-{} V [nA]'.format(label), fontsize = 12, fontweight='bold')

        plt.legend(loc='best')
        plt.savefig('Figures/{}/i_{}_sensor_type.pdf'.format(self.measurement, label))
     
        
        
        plt.clf()          


        

    def plot_itot_histogram(self, df, label):


        batch = list(dict.fromkeys(df['Batch']))
        df = df.loc[df['I_scaled']>10]
        hist, bins, _ = plt.hist(df['I_scaled'], bins=60)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        fig, ax = plt.subplots()
        plt.hist(x=df['I_scaled'], bins= logbins, color='blue')# ax=ax)
        plt.xscale('log')
        ax.set_ylabel('Number of Sensors')
        ax.set_xlabel('Total Current@-{} V [nA]'.format(label), fontsize = 12, fontweight='bold')
        plt.savefig('Figures/{}/i_{}_histogram.png'.format(self.measurement, label))
        plt.clf()




    def plot_number_of_delivered_sensors(self, df):

        df = df.loc[df['Location']=='Hamamatsu']
        pss =np.repeat('PSS', 5960)
    
        s2=np.repeat('2-S', 16200)
        psp = np.repeat('PSP', 6400)
        pss=np.append(pss, s2)
        pss = np.append(pss, psp)
        df1 = pd.DataFrame(pss, columns=['Sensor_type'])
        
        
        pss_ratio = (df.loc[df['Sensor_type']=="PSS"]).shape[0]
        #asa = (df1.loc[df['Sensor_type']=="PSS"]).shape[0]
        
        fig = plt.figure()
        sns.histplot(data=df, x=df['Sensor_type'], hue='Sensor_type', legend=False)
        sns.histplot(data=df1, x='Sensor_type', hue='Sensor_type', edgecolor='k', legend=False )
        plt.xlabel('Sensor Type', fontsize=12)
        plt.ylabel('Number of sensors', fontsize=12)


        pss_ratio = 100*(df.loc[df['Sensor_type']=="PSS"]).shape[0]/5960
        s2_ratio = 100*(df.loc[df['Sensor_type']=="2-S"]).shape[0]/16200
        psp_ratio = 100*(df.loc[df['Sensor_type']=="PSP"]).shape[0]/6400
        fig.legend(labels=["PSP: {}%".format(round(psp_ratio,2)), "2-S: {}%".format(round(s2_ratio,2)), "PSS: {}%".format(round(pss_ratio,2))], title= "Delivered/Total") 
        plt.savefig('figures/number_of_sensors.pdf')

        plt.clf()


    def boxplot(self, df):
    
       plt.clf()
       df = df.loc[df['Location']=='Hamamatsu']
       
       fig, ax = plt.subplots()
       sns.boxplot(data = df, y='Sensor_type', x='I_scaled')
       #xticks = ax.get_xticks()
       #ax.set_xticks(xticks[::len(list(dict.fromkeys(df['Batch'])))//10])  
       #ax.tick_params(axis='x', rotation=90, labelsize = 6)
       plt.show()
       
       plt.clf()
    
       

    def run(self):

        itot_df = SQC_tools.make_dataframe(self, self.measurement, self.iv, ['Sensor', 'Volts', self.iv, 'Temp', 'Run_number'])

        
        i600_df = SQC_tools.filter_dataframe(self, itot_df, 'Volts', 599, 601)
                 
        i800_df = SQC_tools.filter_dataframe(self, itot_df, 'Volts', 799, 801)
        
        

        runs_df = self.runs_dataframe()


        for df, label in ((i600_df, '600'), (i800_df, '800')):
        
           df_final = SQC_tools.sqc_sequence(self, df, runs_df)

           
           df_final = self.scale_current(df_final)
           
         
          # self.make_table_with_outliers(df_final, 'dark_blue', 'IV_sensors_out_of_specs')
           self.plot_batches_over_time(df_final)
           self.boxplot(df_final)
           self.plot_itot_over_time(df_final, label)
           #self.plot_itot_histogram(df_final, label)
           if label=='600':
               self.plot_number_of_delivered_sensors(df_final)
         
           
           
           return df_final





##############################################################################################################################################
##############################################################################################################################################
###############################################                CV                #############################################################
##############################################################################################################################################
##############################################################################################################################################


class CV(runs):


    def __init__(self, cv):

        self.cv = cv
        
        self.config_file = db.read_yml_configuration('SQC_parameters_DB.yml')
        self.measurement = 'SQC'

   

    def find_vfd(self, df):

        list_vfd=[]
        list_sensor=[]
        
        try:
            df[self.cv] = pd.to_numeric(df[self.cv], errors='coerce')
        except Exception as err:
            pass

        df['Run_number']=pd.to_numeric(df['Run_number']) 
        df = df.groupby('Run_number')
        dataframe = [group for _, group in df]
        df_new = pd.DataFrame()
        for i in dataframe:
          try: 
               x, vfd, rho = SQC_tools.analyse_cv(self, i['Volts'].values, (1/i[self.cv]**2).values)
              
          except Exception as err:
            
              vfd=np.nan
          list_sensor.append(i['Sensor'].values[0]) 
          list_vfd.append(vfd) 

        
        df_new['Sensor'] = list_sensor
        df_new['Vfd']= list_vfd

        df_new['Vfd'] = pd.to_numeric(df_new['Vfd']).abs()

        return df_new





    def assign_vfd(self, df):


        df['Volts'] = pd.to_numeric(df['Volts'])
        

        df_new = self.find_vfd(df)
        return df_new       
       
    

    

    def filter_df(self, df):

        df = df.dropna()
        df['Vfd'] = pd.to_numeric(df['Vfd'])
        

        return df




    def plot_vfd_histogram(self, df):


        batch = list(dict.fromkeys(df['Batch']))
        

        fig, ax = plt.subplots()
        plt.hist(x=df['Vfd'], bins= 40, color='blue')# ax=ax)
        ax.set_ylabel('Number of Sensors')
        ax.set_xlabel('Full Depletion Voltage [V]', fontsize = 12, fontweight='bold')
        plt.savefig('Figures/SQC/Vfd_histogram.pdf')


    
    def plot_vfd_over_time(self, df):

     

        batch = list(dict.fromkeys(df['Batch']))

        fig, ax = plt.subplots()
        sns.scatterplot(data = df, x='Batch', y='Vfd', hue='Sensor_type', s=25, linewidth=0, ax=ax) 
         
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::len(batch)//40])  
        ax.tick_params(axis='x', rotation=90, labelsize = 6)
        ax.set(xlabel=None)

        plt.axhline(y = 350, linestyle ='dashed', color='black')
        
        plt.text(26, 380, 'Limit at 350V', color = 'red', style ='italic', fontsize = 12)
       
        ax.set_ylabel('Full Depletion Voltage [V]', fontsize = 12, fontweight='bold')

 
        plt.savefig('Figures/SQC/vfd_evolution.pdf')




    def run(self):

        runs_df = self.runs_dataframe()

        df = SQC_tools.make_dataframe(self, self.measurement, self.cv, ['Sensor', 'Volts', self.cv, 'Run_number'])


        df = SQC_tools.find_location_SQC(self, runs_df, df, 'Sensor')


        new_df= self.assign_vfd(df)
        
        new_df = SQC_tools.find_batch(self, new_df, 'Sensor')
        new_df = SQC_tools.find_wafer_type(self, new_df, 'Sensor')
        
        print(new_df)
        #self.plot_vfd_histogram(new_df)
        self.plot_vfd_over_time(new_df)
        
        return new_df



########################################################################################################################################################################################
########################################################################################################################################################################################
##################################################                Strip parameters               #######################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################




class strip_parameter(runs):



  def __init__(self, parameter):

       self.parameter = parameter
       self.config_file = db.read_yml_configuration('SQC_parameters_DB.yml')
       self.measurement = 'SQC'




  def find_bad_strip(self, df, ylabel, condition):
      
    

    out_of_specs_df = SQC_tools.find_SQC_values_out_of_specs(self, df, self.parameter, self.config_file)
    
    
    if not out_of_specs_df.empty:
    
          SQC_tools.make_html_table(self, out_of_specs_df, 'green_light', '{}_{}_{}_out_of_specs'.format(self.measurement, self.parameter, ylabel))
          if condition=='all_strips':
       
             plt.clf()
            
             sns.histplot(data= out_of_specs_df, x= 'Batch', hue='Location')
             plt.xticks(rotation=90, va='top', fontsize = 6)
 
             plt.ylabel('Strips')
             plt.title('{} - out of specifications'.format(self.parameter))
             plt.savefig('{}_histogram_out_of_specs.png'.format(self.parameter))
             plt.clf()





  def dataframe_with_median(self, df):

      df[self.parameter] = pd.to_numeric(df[self.parameter]).abs()
      
      
      df = df.groupby(['Sensor', 'Run_number', 'Batch', 'Sensor_type', 'Location']).agg(parameter=(self.parameter, 'median'))
      df = df.reset_index()
      df = df.sort_values(by=['Batch'], ascending=True)
    
      df= df.rename(columns={'parameter': self.parameter})

      
      return df





  def normalize_data(self, df):

   
     
     df[self.parameter] = pd.to_numeric(df[self.parameter])

     df[self.parameter] = np.where(df['Sensor_type']=='2-S', df[self.parameter]/self.config_file['SQC_parameters'][self.parameter]['factor_2-S'], df[self.parameter])

     df[self.parameter] = np.where(df['Sensor_type']=='PSS', df[self.parameter]/self.config_file['SQC_parameters'][self.parameter]['factor_PSS'], df[self.parameter])

     if self.parameter=='Idiel':

        df =  self.correct_idiel_only_hephy_data(df)
   
     return df





  def correct_idiel_only_hephy_data(self, df):

      
      wrong_idiel = ['35953', '35715', '35717', '35718']

      df[self.parameter] = df[self.parameter].mask(df['Batch'].apply(lambda x: x in wrong_idiel), df[self.parameter]/1000)
      
      
      return df







  def plot_distribution(self, df, selection, y_label):

    from matplotlib.patches import Rectangle

    df[self.parameter] = df[self.parameter].abs()

  
    fig, ax = plt.subplots()

    upper, lower = SQC_tools.find_outliers(self, df, self.config_file['SQC_parameters'][self.parameter], self.parameter)


    
    hist, bins, patches = plt.hist(df[self.parameter], bins=80, range=(lower, upper), color='blue', edgecolor='black', linewidth=1.2)
    patches[0].set_height(patches[0].get_height() + df.loc[df[self.parameter]<=lower].shape[0])
    patches[0].set_facecolor('green')
    patches[-1].set_height(patches[-1].get_height() + df.loc[df[self.parameter]>=upper].shape[0])
    patches[-1].set_facecolor('yellow')

    ax.set_xlabel('{}'.format(self.config_file['SQC_parameters'][self.parameter]['label']), fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of {}'.format(y_label), fontsize= 12, fontweight='bold')
    extra = Rectangle((0,0), 1, 1, fc='w', fill=False, edgecolor='none', linewidth=0)
    label = 'Specs: {} {}'.format(self.parameter, self.config_file['SQC_parameters'][self.parameter]['annotation'])

    leg = plt.legend([extra], [label], frameon=False, loc='best')

   # plt.xscale('log')
    plt.savefig('Figures/{}/{}_{}.pdf'.format(self.measurement, self.parameter, selection))



  
  def plot_rpoly_over_time(self, df):


        ## plots the scaled current at -600(-800) V over batch number

        batch = list(dict.fromkeys(df['Batch']))

        fig, ax = plt.subplots()
        sns.scatterplot(data = df, x='Batch', y=self.parameter, hue='Sensor_type', s=25, linewidth=0, ax=ax)
         
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::len(batch)//40])  
        ax.tick_params(axis='x', rotation=90, labelsize = 6)
        ax.set(xlabel=None)

        plt.axvline(x=16, linestyle='dashed', color='black')
        plt.axhline(y = 1.0, linestyle ='dashed', color='red')
        
        plt.axhline(y = 2.0, linestyle ='dashed', color='red')
        #plt.text(26, 2000, 'PS-s/PS-p limit', color = 'blue', style ='italic', fontsize = 12)
       
        #plt.text(26, 8500, '2-S limit', color = 'orange', style ='italic', fontsize = 12)
        plt.text(26, 1.8, 'Production Period', style ='italic')


        plt.figtext(0.4, 0.01, r'Time evolution $\longrightarrow$', fontsize=10, fontweight='bold')
        ax.set_ylabel('Polysilicon Resistance [MOhm]', fontsize = 12, fontweight='bold')

        plt.legend(loc='best')
        #plt.show()
        plt.savefig('Figures/SQC/rpoly_evolution.pdf')











  def run(self):

      
      runs_df = self.runs_dataframe()

      df = SQC_tools.make_dataframe(self, self.measurement, self.parameter, ['Sensor', 'Strip', self.parameter, 'Run_number'])
    
      df = SQC_tools.sqc_sequence(self, df, runs_df)  
      
      if self.parameter=='Istrip':
          print(df['Istrip'])
          df['Istrip'] = pd.to_numeric(df['Istrip'])
          print(df.loc[df['Istrip'].abs()>1].to_string())
     
      df = self.normalize_data(df)

      def subfunction(df):
 
      
        for i in ('all_strips','only_medians'):
          
          y_label = 'strips' # used for the y label of the histogram
    
          if i == 'only_medians':
                      
             df = self.dataframe_with_median(df)
           
             y_label = 'sensors'
      
          
          self.find_bad_strip(df, y_label, i)
          
          self.plot_distribution(df, i, y_label)
         
          if self.parameter=='Rpoly':
             self.plot_rpoly_over_time(df)
  
      subfunction(df)
  
      return df


