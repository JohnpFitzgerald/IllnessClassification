# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:06:04 2023

@author: fitzgeraldj
"""
import pandas as pd
import os

def mergeLoop(files):
# Create a DataFrame to store the merged data
  mergedData = pd.DataFrame()   
# Loop through all CSV files in the named directory and concatenate into the new DataFrame
# and apply the filename as a new variable
  for file in os.listdir(files):
    if file.endswith('.csv'):
        # Read the CSV file into a DataFrame and add a new column to identify the file
        data = pd.read_csv(os.path.join(files, file))
        data['id'] = file.split('.')[0]  # Add a new column with the file name as identifier
        
        # Append the DataFrame to the merged_data DataFrame
        mergedData = pd.concat([mergedData, data], ignore_index=True)    
  return mergedData


def mergeDepression():
#merge the DEPRESSION files 
   path = 'C:/mtu/project/depMerge/'    
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/DepReadings.csv', index=False)

def mergeSchizophrenia():
#merge the SCHIZOPHRENIA files 
   path = 'C:/mtu/project/schMerge/'
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/SchReadings.csv', index=False) 
   
def mergeControl():
#merge the CONTROL files
   path = 'C:/mtu/project/conMerge/'
   mergedData = mergeLoop(path)
   mergedData.to_csv('C:/mtu/project/ConReadings.csv', index=False) 
   
   
def merge3Files():
#merge the 3 .csv files    
    Contro = 'C:/mtu/project/ConReadings.csv'
    Schizo = 'C:/mtu/project/SchReadings.csv'
    Depre = 'C:/mtu/project/DepReadings.csv' 
    print("***    Merging multiple files into a single pandas dataframe  ***")
    allData= pd.concat(map(pd.read_csv, [Contro,Schizo,Depre]), ignore_index=True)
    allData.to_csv('C:/mtu/project/AllReadings.csv', index=False)

mergeSchizophrenia() 
mergeDepression()   
mergeControl()
merge3Files()

