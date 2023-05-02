# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:12:29 2023
@author: Jfitz
"""

import keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import seaborn as sns
color = sns.color_palette()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#file='allFeatureReadings.csv'
file = '24HrFeaturesk.csv'
#file = '4HrFeatures.csv'
#file='DayFeatures.csv'
#file='NightFeatures.csv'
df = pd.read_csv(file)


#Prepare the data for modeling
X = df.copy().drop(['id','class','date','category','counter','patientID'],axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2018)


#Create the models:
# Create a pipeline to impute missing values and scale the continuous variables
pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

# Fit the pipeline to the training data and transform the training and testing data
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
    
# calculate descriptive statistics for each feature
desc_stats = df[["f.mean", "f.sd", "f.propZeros","f.kurtosis"]].describe()
print(desc_stats)    

from scipy.stats import ttest_ind


# separate the data into patient and control groups
patient_df = df[df["category"] != "Control"]
control_df = df[df["category"] == "Control"]

# loop through each feature and perform a t-test to compare the distributions between patient and control groups
for feature in ["f.mean", "f.sd", "f.propZeros","f.kurtosis"]:
    patient_data = patient_df[feature]
    control_data = control_df[feature]
    t, p = ttest_ind(patient_data, control_data)
    print("Feature:", feature)
    print("T-statistic:", t)
    print("P-value:", p)
    if p < 0.05:
        print("There is a significant difference in the distribution of this feature between patient and control groups.")
    else:
        print("There is no significant difference in the distribution of this feature between patient and control groups.")


# compute the correlation between "f.mean", "f.sd", and "f.propZeros"
correlation = df[["f.mean", "f.sd", "f.propZeros","f.kurtosis"]].corr()
print(correlation)
# plot the correlation matrix as a heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# add a title to the plot
plt.title("Correlation Matrix")

# show the plot
plt.show()


#----------------------Logistic Regression
logReg = LogisticRegression()

model = logReg

model.fit(X_train, y_train)

# Evaluate the model's performance on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f" Logistic Regression Accuracy: {accuracy}")


import matplotlib.pyplot as plt
import seaborn as sns
# Plot the confusion matrix as a heatmap
label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(confusion_mat, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for LR')
plt.show()

# Extract the classification report values
report = classification_report(y_test, y_pred, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

# Define label names
label_names = ['Control', 'Depression', 'Schizophrenia']
# Create a heatmap
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for Logistic Regression')
plt.show()

#-------------------- Gradient boost

gbm = GradientBoostingClassifier()
# fit the model on the training data
gbm.fit(X_train, y_train)

# predict on the test data
y_pred = gbm.predict(X_test)

# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Gradient Boost Accuracy: {accuracy}")
confusionGB = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(confusionGB, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Gradient Boosting')
plt.show()


# Extract the classification report values
report = classification_report(y_test, y_pred, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

# Create a heatmap
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for Gradient Boost')
plt.show()
#--------------------- Random Forest

# create a random forest classifier object
rfc = RandomForestClassifier()

# fit the model on the training data
rfc.fit(X_train, y_train)

# predict on the test data
y_pred = rfc.predict(X_test)

# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
confusionRF = confusion_matrix(y_test, y_pred)


# Plot the confusion matrix as a heatmap
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(confusionRF, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for RF')
plt.show()


# Extract the classification report values
report = classification_report(y_test, y_pred, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

# Create a heatmap
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for Random Forest Classifier')
plt.show()
#----------------- XG Boost 


# Define the model
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("XGB Accuracy score:", acc_score)


# Plot the confusion matrix as a heatmap
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, cmap="Blues", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for XGB')
plt.show()


# Extract the classification report values
report = classification_report(y_test, y_pred, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

# Create a heatmap
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for XGBoost')
plt.show()

#-----------------------Light GBM


# Define the model
lgb_model = lgb.LGBMClassifier()

lgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lgb_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Light GBM Accuracy score:", acc_score)

# Plot the confusion matrix as a heatmap
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for LGBM')
plt.show()


# Extract the classification report values
report = classification_report(y_test, y_pred, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

# Create a heatmap
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for LightGBM')
plt.show()
#------------------Decision Tree


#  Define the decision tree model
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Decision Tree Accuracy score:", acc_score)

# Plot the confusion matrix as a heatmap
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for RF')
plt.show()


# Extract the classification report values
report = classification_report(y_test, y_pred, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

# Create a heatmap
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for Decision Tree')
plt.show()


