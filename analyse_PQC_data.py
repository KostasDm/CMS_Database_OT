import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import query_PQC_data_from_DB as db
import seaborn as sns
from analysis_tools import PQC_tools
from pretty_html_table import build_table
from matplotlib.patches import Rectangle





########################################################################################################################################################################################
########################################################################################################################################################################################
#################################################          IV parameters          ######################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

class van_der_pauw(PQC_tools):

   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv




   def run(self):

      
       dict_with_dfs= {}
     
       sheet_resistances = ['pstop_vdp', 'strip_vdp', 'poly_vdp']

       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


       # differentiate dataframes by looping over sheet resistances
       for sheet in sheet_resistances:
             
           metadata_sheet_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters'][sheet]['sql_label'])]
           
           df_sheet = df.rename(columns={'R_sheet': sheet})
           
           df_sheet = PQC_tools.pqc_sequence(self, df_sheet, metadata_sheet_df, sheet)
           jsondata = df_sheet[sheet].to_json('{}.json'.format(sheet), orient='records')
           df_sheet[sheet].to_csv('{}.txt'.format(sheet), sep=',')
          
           
           PQC_tools.plot_time_evolution(self, df_sheet, sheet)

           dict_with_dfs[sheet] = df_sheet

       return dict_with_dfs       





class rcont_strip(PQC_tools):

   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv




   def run(self):

      
       dict_with_dfs= {}
     
     

       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])

       
             
       metadata_rcont_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['rcont_strip']['sql_label'])]
           
      #df_sheet = df.rename(columns={'P_edge': })
       #metadata_rcont_df = metadata_rcont_df.loc[metadata_rcont_df['Config'].str.contains('Standard')]
   
            
       df1 = PQC_tools.pqc_sequence(self, df, metadata_rcont_df, 'rcont_strip')
      
       print(df1)
       #df_nor = df1.loc[(df1['p_edge']<10000) & (df1['p_edge']>200)]
       
       plt.hist(df1['rcont_strip'])
       plt.show()
       
       jsondata = df['rcont_strip'].to_json('rcont_strip.json', orient='records')   



       
       
class p_edge(PQC_tools):

   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv




   def run(self):

      
       dict_with_dfs= {}
     
     

       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])

       
             
       metadata_sheet_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['p_edge']['sql_label'])]
           
      #df_sheet = df.rename(columns={'P_edge': })
       metadata_sheet_df = metadata_sheet_df.loc[metadata_sheet_df['Config'].str.contains('Standard')]
       df_sheet = df.rename(columns={'R_sheet': 'p_edge'})
            
       df1 = PQC_tools.pqc_sequence(self, df_sheet, metadata_sheet_df, 'p_edge')
       df_nor = df1.loc[(df1['p_edge']<10000) & (df1['p_edge']>200)]
       
       plt.hist(df_nor['p_edge'])
       plt.show()
       
       jsondata = df_nor['p_edge'].to_json('p_edge.json', orient='records')

          
           
      

       return dict_with_dfs  




       
class linewidth(PQC_tools):



   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      
       
       linewidth_structures = ['linewidth_strip', 'linewidth_pstop'] 

     
       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])



       for line in linewidth_structures:
             
           metadata_lwth_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters'][line]['sql_label'])]

           df_lwth = df.rename(columns={'Linewidth': line})
           

           df_lwth = PQC_tools.pqc_sequence(self, df_lwth, metadata_lwth_df, line)
           

           PQC_tools.plot_time_evolution(self, df_lwth, line)




class tox():



   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      

      df_tox = tools.make_dataframe('PQC', self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


      metadata_df = tools.make_dataframe('PQC', self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])



     
      metadata_tox_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Tox']['sql_label'])]
     
      df_tox = tools.pqc_sequence(df_tox, metadata_tox_df, 'Halfmoon', 'Tox')


      SQC_tools.plot_time_evolution(self, df_tox, 'Dox', 'Halfmoon_type', self.config_file['PQC_parameters']['Tox']['ylabel'])











class vdp_bulk(PQC_tools):



   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      


       df = PQC_tools.make_dataframe(self,  self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_bulk_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['vdp_bulk']['sql_label'])]
           

       df_vdp_bulk = PQC_tools.pqc_sequence(self, df, metadata_bulk_df, 'vdp_bulk')
           
          
       df_vdp_bulk= df_vdp_bulk.loc[df_vdp_bulk['vdp_bulk']<self.config_file['PQC_parameters']['vdp_bulk']['upper']]


       df_vdp_bulk= df_vdp_bulk.loc[df_vdp_bulk['vdp_bulk']>self.config_file['PQC_parameters']['vdp_bulk']['lower']]

       #####
       df_standard = df_vdp_bulk.loc[df_vdp_bulk['Config'].str.contains('Standard')]
       df_standard = df_standard.drop_duplicates(subset=['Halfmoon', 'vdp_bulk'])

       df_perugia = df_standard.loc[df_standard['Location'].str.contains('Perugia')]
       print(df_perugia.to_string())
       df_standard =df_standard.loc[~df_standard['Location'].str.contains('Perugia')]
       #####

       
       df_2 = df_standard.loc[df_standard['Halfmoon'].str.contains('WW')]
    
       PQC_tools.plot_time_evolution(self, df_standard, 'vdp_bulk') #self.config_file['PQC_parameters']['vdp_bulk']['ylabel']


       return df_standard




class I_surf(PQC_tools):



   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_isurf_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Isurf']['sql_label'])]
           

       df_isurf = PQC_tools.pqc_sequence(self, df, metadata_isurf_df, 'Isurf')
       jsondata = df_isurf['Isurf'].to_json('Isurf.json', orient='records')  
       df_isurf['Isurf'].to_csv('Isurf.txt', sep=',')       
          

       PQC_tools.plot_time_evolution(self, df_isurf, 'Isurf')





class S0(PQC_tools):



   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_S0_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['S0']['sql_label'])]
           

       df_S0 = PQC_tools.pqc_sequence(self, df, metadata_S0_df, 'S0')
           
          
       jsondata = df_S0['S0'].to_json('S0.json', orient='records')
       df_S0['S0'].to_csv('S0.txt', sep=',')
       PQC_tools.plot_time_evolution(self, df_S0, 'S0')




############################################################################################################
############################################################################################################
################################## CV ######################################################################

class Vfb(PQC_tools):



   def __init__(self, cv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_vfb_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Vfb']['sql_label'])]
           

       df_vfb = PQC_tools.pqc_sequence(self, df, metadata_vfb_df, 'Vfb')
           
       jsondata = df_vfb['Vfb'].to_json('Vfb.json', orient='records')
       df_vfb['Vfb'].to_csv('Vfb.txt', sep=',')
       PQC_tools.plot_time_evolution(self, df_vfb, 'Vfb')




class Nox(PQC_tools):



   def __init__(self, cv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_nox_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Nox']['sql_label'])]
           

       df_nox = PQC_tools.pqc_sequence(self, df, metadata_nox_df, 'Nox')
           
       jsondata = df_nox['Nox'].to_json('Nox.json', orient='records')
       df_nox['Nox'].to_csv('Nox.txt', sep=',')
           
       PQC_tools.plot_time_evolution(self, df_nox, 'Nox')      
     



     
class Tox(PQC_tools):



   def __init__(self, cv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_tox_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Tox']['sql_label'])]
           

       df_tox = PQC_tools.pqc_sequence(self, df, metadata_tox_df, 'Tox')
           
       jsondata = df_tox['Tox'].to_json('Tox.json', orient='records')
       df_tox['Tox'].to_csv('Tox.txt', sep=',')
       PQC_tools.plot_time_evolution(self, df_tox, 'Tox')       
 


 
class Dox(PQC_tools):



   def __init__(self, cv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_dox_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Dox']['sql_label'])]
           

       df_dox = PQC_tools.pqc_sequence(self, df, metadata_dox_df, 'Dox')
           
       
       PQC_tools.plot_time_evolution(self, df_dox, 'Dox')
       
       
       
              
       
class Diode_bulk(PQC_tools):



   def __init__(self, cv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv




   def find_bulk_resistivity(self, df):

       thickness = 290*1e-4
       e0 = 8.85*1e-14
       e_r = 11.68
       mu_h = 450 

       df['Diode_bulk'] = (thickness*thickness)/(2*e0*e_r*mu_h*df['Diode_Vfd']*1000)

       return df

      


   def run(self):

      
       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])
      

       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])
       df = PQC_tools.pqc_sequence(self,  df, metadata_df, 'Diode_Vfd')
       

            
       df_diode_bulk = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Diode_Vfd']['sql_label'])]
       df_diode_bulk['Diode_Vfd'] = pd.to_numeric(df_diode_bulk['Diode_Vfd'])
           

       
        
       
       df_diode_bulk= df_diode_bulk.loc[df_diode_bulk['Diode_bulk']<self.config_file['PQC_parameters']['Diode_bulk']['upper']]

       df_diode_bulk= df_diode_bulk.loc[df_diode_bulk['Diode_bulk']>self.config_file['PQC_parameters']['Diode_bulk']['lower']]
        

       df_diode_bulk = df_diode_bulk.drop_duplicates(subset=['Halfmoon', 'Diode_bulk'])
       
       df_perugia = df_diode_bulk.loc[df_diode_bulk['Location'].str.contains('Perugia')]
       df_diode_bulk2 = df_diode_bulk.loc[~df_diode_bulk['Location'].str.contains('Perugia')]
       
       print(df_perugia.to_string())

       PQC_tools.plot_time_evolution(self, df_perugia, 'Diode_bulk') # self.config_file['PQC_parameters']['Diode_bulk']['ylabel'])
       
       
       return df_diode_bulk
       




class Diode_bulk_raw_data(Diode_bulk):



   def __init__(self, cv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv




   def find_bulk_resistivity(self, df):

       thickness = 290*1e-4
       e0 = 8.85*1e-14
       e_r = 11.68
       mu_h = 450 

       df['Diode_bulk'] = (thickness*thickness)/(2*e0*e_r*mu_h*df['Diode_Vfd']*1000)

       return df


   def find_active_thickness(self, df):
    
      df['Area'] = np.nan
      df['Area'] = df['Area'].mask(df['Halfmoon'].str.contains('2-S'), 92.13*1e-4).mask(df['Halfmoon'].str.contains('PSS'), 45.30*1e-4).mask(df['Halfmoon'].str.contains('PSP'), 45.30*1e-4)
      A_2S = 92.13 #cm**2
      A_PSS = 45.30
      A_PSP = 45.30
      
      return df
      
      


   def df_from_raw_data(self):


       df = PQC_tools.make_dataframe(self, 'CV_raw_Data', ['Condition_number', 'Part_ID', 'Halfmoon', 'Volts', 'Cap', 'Bias_Current'])

       return df        


   def find_capacitance(self, df, df2):
   
       df = df.groupby('Part_ID')
       dataframe = [group for _, group in df]

       df_new = pd.DataFrame()
       for i in dataframe:
         
          a = df2.loc[df2['Halfmoon'].isin(i["Halfmoon"])]
          a['Diode_Vfd'] = pd.to_numeric(a['Diode_Vfd'])
          i['Volts'] = pd.to_numeric(i['Volts'])
          if not a.empty:
              
  
             i['diff'] = (i['Volts']-a['Diode_Vfd'].values[0]).abs()
            
             cap = (i['Cap'].loc[i['diff'] == i['diff'].min()]).values[0]
             print((11.68*(8.854*1e-12)* 6.25*1e-6)/(cap*1e-12))
           
          #i['Halfmoon'].values[0]])])# & ((df['Cap'] - df2['Diode_Vfd'].abs())<5)])
           
       #print(i)
   


   def find_vfd(self, df):

       list_hf = []
       list_vfd = []
       esi=11.68
       e0 = 8.854*1e-12

       df['Part_ID'] = pd.to_numeric(df['Part_ID'])
       df['Cap'] = df['Cap']*1e-12
       df = df.groupby('Part_ID')
       dataframe = [group for _, group in df]

       df_new = pd.DataFrame()
       for i in dataframe:
           i = i.sort_values(by=['Volts'])
           try:
               x, vfd, rho = PQC_tools.analyse_cv(self, i['Volts'].values, (1/i['Cap']**2).values) 
              
           except Exception as err:
               vfd =np.nan
               rho = np.nan

           list_hf.append(i['Halfmoon'].values[0])
           list_vfd.append(vfd)
       
           a = i.loc[i['Volts'].between(round(vfd, 2), round(vfd, 2)+5, inclusive=False)]
          # if a.shape[0]>0:
            # d_active = a['Cap'].values[0]
            # print(d_active)

       df_new['Halfmoon'] = list_hf
       df_new['Diode_Vfd'] = list_vfd

       df_new['Diode_Vfd'] = pd.to_numeric(df_new['Diode_Vfd']).abs()
       
       return df_new
       
       
       
       
       
   def run(self):

      
       df_raw = self.df_from_raw_data()
       
      
       
       
      # print(df_raw.loc[df_raw['Halfmoon']=='37907_002_2-S_HM_EE'].to_string())





       
       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])
       
       #df = PQC_tools.merge_PQC_dataframes(self, df, metadata_df, 'Diode_bulk')
       #bf = PQC_tools.merge_PQC_dataframes(self, df_raw, metadata_df, 'Cap')
       
       #df_raw = bf.loc[bf['Parameter'].str.contains('DIODE_HALF')]
       df_vfd = Diode_bulk.run(self)
       
       df_vfd = self.find_active_thickness(df_vfd)
       df_vfd['Diode_Vfd'] = pd.to_numeric(df_vfd['Diode_Vfd'])
      
       df_raw['Volts'] = pd.to_numeric(df_raw['Volts']).abs()

       df_raw['Cap'] = pd.to_numeric(df_raw['Cap'])
       #df_vfd = df_vfd.drop_duplicates(subset=['Halfmoon', 'Diode_Vfd'])
    
       self.find_capacitance(df_raw, df_vfd)
       #df_new = self.find_vfd(df_raw)
       #df_new = self.find_bulk_resistivity(df_new)
  

###########################################################################################################
###########################################################################################################
########################## FET ############################################################################

  
            
class Vth(PQC_tools):



   def __init__(self, fet):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.fet = fet



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.fet, self.config_file['PQC_tables'][self.fet]['dataframe_headers'])


       metadata_df = PQC_tools.make_dataframe(self, self.metadata, self.config_file['PQC_tables'][self.metadata]['dataframe_headers'])


             
       metadata_fet_df = metadata_df.loc[metadata_df['Parameter'].str.contains(self.config_file['PQC_parameters']['Vth']['sql_label'])]
           

       df_fet = PQC_tools.pqc_sequence(self, df, metadata_fet_df, 'Vth')
           
      
       jsondata = df_fet['Vth'].to_json('Vth.json', orient='records')
       df_fet['Vth'].to_csv('Vth.txt', sep=',')
       PQC_tools.plot_time_evolution(self, df_fet, 'Vth')   
        

#############################################################################################################################################################################################
#############################################################################################################################################################################################
######################################################################


def run_alles():
  class_dictionary = {'IV': ['rcont_strip', 'p_edge', 'van_der_pauw', 'linewidth', 'I_surf', 'S0', 'vdp_bulk'], 'CV': ['Vfb', 'Nox', 'Tox', 'Dox'], 'FET': ['Vth']} # 'Diode_bulk''CV': ['Diode_bulk']} 'CV': ['Diode_bulk_raw_data']
  df_list=[]


  for  class_ in class_dictionary.keys():
    for structure in class_dictionary[class_]:
          
          v = globals()[structure](str(class_))
      
          #p = v(str(class_)) #    IV(str(n)) 
          df = v.run()
          if structure=='vdp_bulk' or structure=='Diode_bulk':
              df_list.append(df)


if __name__ == "__main__":
  run_alles()
'''
df_vdp=df_list[0]
df_diode = df_list[1]

df_vdp2 = df_vdp[['Halfmoon', 'Position', 'vdp_bulk']]

df_final = pd.merge(df_diode, df_vdp2, on=['Halfmoon'], how='inner')
#print(df_final.loc[df_final['Batch'].str.contains('38868')].to_string())
#df_final = df_final.loc[df_final['Diode_bulk']<8]


corr = df_final['Diode_bulk'].corr(df_final['vdp_bulk'])
print(corr)

plt.clf()

sns.lmplot(x='Diode_bulk', y= 'vdp_bulk',  data=df_final, ci=None)
plt.ylabel('VdP \u03C1 [k\u03A9*cm]', fontsize=12)
plt.xlabel('Diode \u03C1 [k\u03A9*cm]', fontsize=12)
plt.text(3, 7, 'Correlation Coeff.: {}'.format(round(corr, 3)))
#plt.xlim(1,8)
#plt.ylim(1,8)
plt.savefig('figures/correlation_rho.pdf')
'''
