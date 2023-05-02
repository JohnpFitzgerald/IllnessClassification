# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:30:08 2023

@author: Jfitz
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# =============================================================================
#3 files merged 
file = 'AllReadings.csv'
data = pd.read_csv(file)
# convert the "date" column to a datetime object
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = pd.to_datetime(data['timestamp'])    
#examine data
print(data.dtypes)
print(f"no of records:  {data.shape[0]}")
print(f"no of variables: {data.shape[1]}")
print((data['id'].nunique()))
# want to drop cases that dont have 24 hours of returns                               
#aggregate date and hour and include data for 24 hour period only

#count the number of days recorded for each participant
# IE the number of unique dates for each id
date_counts = data.groupby("id")["date"].nunique()
# Write the results to a text file
with open("ParticipantDayCounts.txt", "w") as file:
    for index, value in date_counts.items():
        file.write(f"{index}: {value}\n")
# Print the results
#print(date_counts)


data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute + data['hour'] * 60
aggr = data.groupby(['date','hour']).agg({'activity': 'sum'}).reset_index()
aggr = aggr[(aggr['hour'] >= 0) & (aggr['hour'] <= 23)]
counted = aggr.groupby('date').agg({'hour' : 'count'}).reset_index()
counted = counted[counted['hour'] == 24]

final = pd.merge(data, counted[['date']], on='date', how='inner')

counts = final.groupby(['id', 'date']).count()
valid_groups = counts[counts['activity'] == 1440].reset_index()[['id', 'date']]
final = final.merge(valid_groups, on=['id', 'date'])


#print(final.head())
#####Create text categories - for visualisations
def newId(idVal):
    if idVal[:5] == 'condi':
        return 'Depressive'
    elif idVal[:5] == 'patie':
        return 'Schizophrenic'
    elif idVal[:5] == 'contr':
        return 'Control'
    else:
        return '*UNKNOWN*'
  
final['category'] = final['id'].apply(newId)

if '*UNKNOWN*' in final['category'].values:
    print("unknowns found") 
else:
    print("All 24 hours have a category")   

#add a counter for the 3 categories (visualisations)    
final['counter'] = final.groupby('category').cumcount() + 1 

 
# create a patient dictionary to map each unique ID 
patient = {id:index + 1 for index, id in enumerate(data['id'].unique())}
# map the ID col to the pateientID using the dictionary values
final['patientID'] = final['id'].map(patient)

num_records = len(final)
print(f"Number of records in dataframe: {num_records}")

#Ensure 24 hours of entries for all particpants 
if num_records % 1440 == 0:
   print("Number of records is divisible by 1440 with no remainder")
else:
   print("Number of records is NOT divisible by 1440")

#create a segment category based on the time
def seg(hr):
    if hr < 4:
        return '00:00-04:00'
    elif hr > 3 and hr < 8:
        return '04-08:00'
    elif hr > 7 and hr < 12:
        return '08-12:00'
    elif hr > 11 and hr < 16:
        return '12-16:00'
    elif hr > 15 and hr < 20:
        return '16-20:00'
    elif hr > 19 and hr < 24:
        return '20-24:00'    
    else:
        return '*UNKNOWN*'
  
final['segment'] = final['hour'].apply(seg)

#all data is segmented?
if '*UNKNOWN*' in final['segment'].values:
    print("unknowns segments found") 
else:
    print("All ids have segments")   




#####################create features for 24 hour data########################

# create features for 24 hour data
grouped = final.groupby(['id','date'])
newData = grouped.agg({'activity': ['mean', 'std', lambda x: (x == 0).mean()]})
newData = newData.reset_index()

newData.columns = ['id','date','f.mean', 'f.sd', 'f.propZeros']
 
newData['class'] = newData['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
newData = newData[['id','date','f.mean','f.sd','f.propZeros','class']]
discard = newData.loc[((newData['f.mean'] == 0) & (newData['f.sd'] == 0))]
discard.to_csv('C:/mtu/project/removedCases.csv', index=False)
newData = newData.loc[~((newData['f.mean'] == 0) & (newData['f.sd'] == 0))]
print((newData['id'].nunique()))

#create features for 4 hourly data
grouped = final.groupby(['id','date','segment'])
segmented = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
segmented = segmented.reset_index()
segmented.columns = ['id','date','segment','f.mean', 'f.sd', 'f.propZeros']
 
segmented['class'] = segmented['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
segmented = segmented[['id','date','segment','f.mean','f.sd','f.propZeros','class']]
segmented= segmented.loc[~((segmented['f.mean'] == 0) & (segmented['f.sd'] == 0))]
print((segmented['id'].nunique()))


# Create a daytime and nighttime dataframe
Daysegmented = final.loc[~(final['segment'].isin(['00:00-04:00','04-08:00','20-24:00']))]
Nightsegmented = final.loc[~(final['segment'].isin(['08-12:00','12-16:00','16-20:00']))]
#create features for daytime data 8am - 8pm hour data
grouped = Daysegmented.groupby(['id','date'])
dayData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
dayData = dayData.reset_index()
dayData.columns = ['id','date','f.mean', 'f.sd', 'f.propZeros'] 
dayData['class'] = dayData['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
dayData = dayData[['id','date','f.mean','f.sd','f.propZeros','class']]
dayData = dayData.loc[~((dayData['f.mean'] == 0) & (dayData['f.sd'] == 0))]
print((dayData['id'].nunique()))


#create features for night data 8pm - 8am hour data
grouped = Nightsegmented.groupby(['id','date'])
nightData = grouped.agg({'activity': ['mean','std', lambda x: (x == 0).mean()]})
nightData = nightData.reset_index()
nightData.columns = ['id','date','f.mean', 'f.sd', 'f.propZeros'] 
nightData['class'] = nightData['id'].str[:5].apply(lambda x: 1 if x == 'condi' else (0 if x == 'contr' else 2))
nightData = nightData[['id','date','f.mean','f.sd','f.propZeros','class']]
nightData = nightData.loc[~((nightData['f.mean'] == 0) & (nightData['f.sd'] == 0))]
print((nightData['id'].nunique()))

# =============================================================================
def newId(idVal):
     if idVal[:5] == 'condi':
         return 'Depressive'
     elif idVal[:5] == 'patie':
         return 'Schizophrenic'
     elif idVal[:5] == 'contr':
         return 'Control'
     else:
         return '*UNKNOWN*'
   
newData['category'] = newData['id'].apply(newId)
segmented['category'] = segmented['id'].apply(newId) 
dayData['category'] = dayData['id'].apply(newId) 
nightData['category'] = nightData['id'].apply(newId) 

if '*UNKNOWN*' in newData['category'].values:
    print("unknowns found") 
else:
    print("All 24 hours have a category")   
if '*UNKNOWN*' in dayData['category'].values:
    print("unknowns found") 
else:
    print("All daytime data have a category") 
if '*UNKNOWN*' in nightData['category'].values:
    print("unknowns found") 
else:
    print("All nighttime data have a category") 
if '*UNKNOWN*' in segmented['category'].values:
    print("unknowns found") 
else:
    print("All segment data has a category")       
newData['counter'] = newData.groupby('category').cumcount() + 1
segmented['counter'] = segmented.groupby('category').cumcount() + 1
dayData['counter'] = dayData.groupby('category').cumcount() + 1
nightData['counter'] = nightData.groupby('category').cumcount() + 1
# 
# # create a patient dictionary to map each unique ID 
patient = {id:index + 1 for index, id in enumerate(newData['id'].unique())}
patient = {id:index + 1 for index, id in enumerate(dayData['id'].unique())}
patient = {id:index + 1 for index, id in enumerate(nightData['id'].unique())}
patient = {id:index + 1 for index, id in enumerate(segmented['id'].unique())}
# # map the ID col to the pateientID using the dictionary values
newData['patientID'] = newData['id'].map(patient)
segmented['patientID'] = segmented['id'].map(patient)
dayData['patientID'] = dayData['id'].map(patient)
nightData['patientID'] = nightData['id'].map(patient)
# =============================================================================

#print(newData)    
#print(segmented) 
#print(dayData)
#print(nightData)
print("***  All 3 groups Baseline input file created for 24 hr of data only ***")
print("***  Features created for 4 hourly segments/ Daytime and Nighttime ****")


#create output csvs for use in later scripts
newData.to_csv('C:/mtu/project/24HrFeatures.csv', index=False)
segmented.to_csv('C:/mtu/project/4HrFeatures.csv', index=False)
nightData.to_csv('C:/mtu/project/NightFeatures.csv', index=False)
dayData.to_csv('C:/mtu/project/DayFeatures.csv', index=False)

# =============================================================================
# Plots
#-----------------------------------------------------------------------------

# Plot with comaparison of particpants from each caetgory with similar demographics
##################### compare 3 females 40-44 ###################################
extract18 = final.query("id == 'condition_18'").head(18720)   #id 64 - 20160 rows 14 days
extract8 = final.query("id == 'control_8'").head(18720)       #id 31 - 27360 rows 19
extractp8 = final.query("id == 'patient_8'").head(18720)      #id 53 - 27360 rows 19

# Concatenate the data into a new dataframe
extract = pd.concat([extract18, extract8, extractp8], keys=['control_8','condition_18', 'patient_8'])
#print(extract18)
#print(extract8)
#print(extractp8)
# Compute the mean activity by category and hour
grouped = extract.groupby(['category', 'hour'])['activity'].mean().reset_index()


# Pivot the data to a wide format
pivoted = grouped.pivot(index='hour', columns='category', values='activity')

# Plot the data
plt.plot(pivoted.index, pivoted['Control'], label='Control 8')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive 18')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic 8')
plt.xticks(range(24), [f'{h:02d}' for h in range(24)])
plt.xlabel(' 24 hour cycle')
plt.ylabel('Average activity rate')
plt.title('Average 24 hour activity of female aged 40-44 from each group')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('females40-44.png', dpi=300)
plt.show()

#boxplot
Condition18 = extract18.groupby(['category', 'hour'])['activity'].mean()
Control8 = extract8.groupby(['category', 'hour'])['activity'].mean()
Patient8 = extractp8.groupby(['category', 'hour'])['activity'].mean()
activ = [Condition18,Control8,Patient8]
fig, ax = plt.subplots()
ax.boxplot(activ)
ax.set_xticklabels(['Depressive 18','Control 8','Schizophrenic 8'])
ax.set_ylabel('Daily average activity')
ax.set_title("Daily average activities of female aged 40-44 from each group")
plt.show() 


# =======================3 females 50-59=====================================
extract5 = final.query("id == 'condition_5'").head(18720)    #id73 - 20160 rows 14 days
extract27 = final.query("id == 'control_27'").head(18720)    #id20 - 18720 rows 13 days
extract12 = final.query("id == 'patient_12'").head(18720)    #id36 - 18720 rows 13 days

# Concatenate the data into a new dataframe
extract = pd.concat([extract5, extract27, extract12], keys=['control_27','condition_5', 'patient_12'])
#print(extract5)
#print(extract27)
#print(extract12)
# Compute the mean activity by category and hour
grouped = extract.groupby(['category', 'hour'])['activity'].mean().reset_index()

# Pivot the data to a wide format
pivoted = grouped.pivot(index='hour', columns='category', values='activity')

# Plot the data
plt.plot(pivoted.index, pivoted['Control'], label='Control 27')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive 5')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Shizophrenic 12')



plt.xticks(range(24), [f'{h:02d}' for h in range(24)])
plt.xlabel(' 24 hour cycle')
plt.ylabel('Average activity rate')
plt.title('Average 24 hour activity of female aged 50-59 from each group')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('females50-59.png', dpi=300)
plt.show()

#boxplot
Condition5 = extract5.groupby(['category', 'hour'])['activity'].mean()
Control27 = extract27.groupby(['category', 'hour'])['activity'].mean()
Patient12 = extract12.groupby(['category', 'hour'])['activity'].mean()
activ = [Condition5,Control27,Patient12]
fig, ax = plt.subplots()
ax.boxplot(activ)
ax.set_xticklabels(['Depressive 5','Control 27','Schizophrenic 12'])
ax.set_ylabel('Daily average activity')
#ax.set_xlabel('Patient 22 v Control 25')
ax.set_title("Daily average activities of female aged 50-59 from each group")
plt.show() 


# ======================= 3 males aged 30-34 =======================================
extract20 = final.query("id == 'condition_20'").head(18720)  #id67 24480 17 days
extract5 = final.query("id == 'control_5'").head(18720)      #id28 46080  32 days
extract11 = final.query("id == 'patient_11'").head(18720)                 #id35 18720 13 days

# Concatenate the data into a new dataframe
extract = pd.concat([extract20, extract5, extract11], keys=['control_5', 'condition_20', 'patient_11'])
#print(extract20)
#print(extract5)
#print(extract11)
# Compute the mean activity by category and hour
grouped = extract.groupby(['category', 'hour'])['activity'].mean().reset_index()

# Pivot the data to a wide format
pivoted = grouped.pivot(index='hour', columns='category', values='activity')

plt.plot(pivoted.index, pivoted['Control'], label='Control 5')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive 20')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic 11')

plt.xticks(range(24), [f'{h:02d}' for h in range(24)])
plt.xlabel(' 24 hour cycle')
plt.ylabel('Average activity rate')
plt.title('Average 24 hour activity of male aged 30-34 from each group')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('males30-34.png', dpi=300)
plt.show()

Condition20 = extract20.groupby(['category', 'hour'])['activity'].mean()
Control5    = extract5.groupby(['category', 'hour'])['activity'].mean()
Patient11   = extract11.groupby(['category', 'hour'])['activity'].mean()
activ = [Condition20,Control5,Patient11]
fig, ax = plt.subplots()

ax.boxplot(activ)



ax.set_xticklabels(['Depressive 20','Control 5','Schizophrenic 11'])
ax.set_ylabel('Daily average activity')
ax.set_title("Daily average activities of male aged 30-34 from each group")
plt.show()
# =============================================================================
############# plot of 24 hour averages #################################
#plot of averages of activity from midnight to midnight by hour
grouped = final.groupby(['category', 'hour'])['activity'].mean().reset_index()
pivoted = grouped.pivot(index='hour', columns='category', values='activity')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

plt.xticks(range(24), [f'{h:02d}' for h in range(24)])

plt.xlabel(' 24 hour period hourly')
plt.ylabel('Average of daily Activity')
plt.title('Average Activity midnight to midnight per hour by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('AverageActivityPerHour.png', dpi=300)
plt.show()


#plot of 24 hours by minute
grouped = final.groupby(['category', 'minute'])['activity'].mean().reset_index()
pivoted = grouped.pivot(index='minute', columns='category', values='activity')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')

plt.xlabel(' 24 hour period in minutes')
plt.ylabel('Average Activity')
plt.title('Average Activity per minute by Category')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('AverageActivityPerMinute.png', dpi=300)
plt.show()
#################### Plot of averages 4 hour segments ########################
#plot of 4 hourly averages 
grouped = segmented.groupby(['category', 'segment'])['f.mean'].mean().reset_index()
pivoted = grouped.pivot(index='segment', columns='category', values='f.mean')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')
plt.rcParams["figure.autolayout"] = True
plt.xlabel(' 24 hour period 4 hour segments')
plt.ylabel('Mean Activity of averages')
plt.title('Mean Activity in 4 hourly segments over 24 hours')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('MeanActivityPer4hrSegement.png', dpi=300)
plt.show()

# plot of sd averages every 4 hours
grouped = segmented.groupby(['category', 'segment'])['f.sd'].mean().reset_index()
pivoted = grouped.pivot(index='segment', columns='category', values='f.sd')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')
plt.rcParams["figure.autolayout"] = True
plt.xlabel(' 24 hour period 4 hour segments')
plt.ylabel('Standard Deviation Activity')
plt.title('SD  Activity in 4 hourly segments over 24 hours')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('SD-ActivityPer4hrSegement.png', dpi=300)
plt.show()

#plot of propZeros every 4 hours
grouped = segmented.groupby(['category', 'segment'])['f.propZeros'].mean().reset_index()
pivoted = grouped.pivot(index='segment', columns='category', values='f.propZeros')
plt.plot(pivoted.index, pivoted['Control'], label='Control')
plt.plot(pivoted.index, pivoted['Depressive'], label='Depressive')
plt.plot(pivoted.index, pivoted['Schizophrenic'], label='Schizophrenic')
plt.rcParams["figure.autolayout"] = True
plt.xlabel(' 24 hour period 4 hour segments')
plt.ylabel('Proportion of Zero values Activity')
plt.title('Prop of Zero vals  Activity in 4 hourly segments over 24 hours')
plt.legend()
plt.gcf().set_size_inches(12, 6)
plt.savefig('PropZeros-ActivityPer4hrSegement.png', dpi=300)
plt.show()