import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml
import os




parameters = ['CV']#'IV', 'CV', 'FET']




def read_yml_configuration(yml_file):

  with open(yml_file, 'r') as f:
     conf = yaml.load(f, Loader=yaml.FullLoader)

  return conf





def query_data_from_DB(table_header_list, sql_table_prefix, parameter):

  
  print('The query of {} PQC data from the CMS DB is gonna take a while'.format(parameter))

 
  if parameter == 'FET':
       
       p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select d.{}, d.{}, d.{}, d.{} from trker_cmsr.c{} d".format(*table_header_list, sql_table_prefix)], capture_output=True)

  else:
       
       p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select d.{}, d.{}, d.{}, d.{}, d.{}, d.{}, d.{}, d.{} from trker_cmsr.c{} d".format(*table_header_list, sql_table_prefix)], capture_output=True)
  
       
  answer = p1.stdout.decode().splitlines()
  print(answer)
  print('Query of {} PQC data is complete'.format(parameter))
  
  return answer


 


  
def query_metadata_table_from_DB():


  print('Querying the metadata information from the CMS DB')

  p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
      "select d.CONDITION_DATA_SET_ID, d.PART_ID, d.PART_NAME_LABEL, d.KIND_OF_HM_FLUTE_ID, d.KIND_OF_HM_STRUCT_ID, d.KIND_OF_HM_CONFIG_ID, d.KIND_OF_HM_SET_ID, d.FILE_NAME  from trker_cmsr.c8920 d"], capture_output=True)

  answer = p1.stdout.decode().splitlines()
  
  print('Query of metadata information is complete')

  return answer




 
  
def save_DB_table_as_json(answer_from_DB, filename):

  with open('PQC_data/{}.json'.format(filename), 'w') as file:
    json.dump(answer_from_DB[1:], file)




    
def make_list_from_json(file):

   with open(file, 'r') as file:
       data = json.load(file)
       
   return data

   
   

def process_list_with_data(data):

   split_data = [] 
    
   for i in data:
      split_data.append(i.split(','))
      
    
   return split_data
   



  



def generate_json_with_data(pqc_parameters):

  for i in parameters:
    
      headers = pqc_parameters[str(i)]['table_headers']
      sql_table = pqc_parameters[str(i)]['sql_table_prefix']
      answer_from_DB = query_data_from_DB(headers, sql_table, str(i))
      save_DB_table_as_json(answer_from_DB, str(i))

  metadata_answer = query_metadata_table_from_DB() 
  save_DB_table_as_json(metadata_answer, 'metadata')




def run():

 
  try:
    os.mkdir("PQC_data")

  except FileExistsError:
    print("Directory data/ already exists")



  configuration = read_yml_configuration('PQC_parameters_DB.yml')
  pqc_parameters = configuration['PQC_tables']

  generate_json_with_data(pqc_parameters)
  
 



if __name__=="__main__":

    run()   
  
 
