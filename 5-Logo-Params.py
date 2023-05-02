import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, LeavePGroupsOut, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_score, f1_score
import itertools
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
color = sns.color_palette()

#file='allFeatureReadings.csv'
file = '4HrFeaturesk.csv'
#file='DayFeaturesk.csv'
#file='NightFeaturesk.csv'
#file = '24HrFeaturesk.csv'
df = pd.read_csv(file)



X = df[['f.mean', 'f.sd', 'f.propZeros', 'f.kurtosis']]
y = df['class'].values
groups = df['patientID'].values

#  classifiers
rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=2018)
dtc = DecisionTreeClassifier(max_depth=10, random_state=2018)
xgbc = xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=2018)
# pipeline to impute missing values and scale the continuous variables
pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())


logo = LeaveOneGroupOut()

#predicted and actual labels
predicted_labels = np.array([])
true_labels = np.array([])


for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the pipeline a
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    # Balance the target 
    smote = SMOTE(random_state=2018)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    rfc.fit(X_train, y_train)

 
    predicted_labels = np.append(predicted_labels, rfc.predict(X_test))
    true_labels = np.append(true_labels, y_test)


accuracy = accuracy_score(true_labels, predicted_labels)
print("Random Forest Accuracy:", accuracy)
cmrf = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix Random Forest:")
print(cmrf)
crrf = classification_report(true_labels, predicted_labels)
print("Classification Report Random Forest:")
print(crrf)

cmrf = confusion_matrix(true_labels, predicted_labels)
#print(cmdt)
#print("Decision Tree - Classification Report:")
#print(classification_report(y_test, dtc_pred))
cmrf = np.array(cmrf)
label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(cmrf, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest')
plt.show()

report = classification_report(true_labels, predicted_labels, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])


label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for Random Forest')
plt.show()



for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    # Balance the target 
    smote = SMOTE(random_state=2018)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    dtc.fit(X_train, y_train)


    predicted_labels = np.append(predicted_labels, dtc.predict(X_test))
    true_labels = np.append(true_labels, y_test)


accuracy = accuracy_score(true_labels, predicted_labels)
print("Decision Tree Accuracy:", accuracy)
cmdt = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix Decision Tree:")
print(cmdt)
cr = classification_report(true_labels, predicted_labels)
print("Classification Report Decision Tree:")
print(cr)

cmdt = confusion_matrix(true_labels, predicted_labels)
cmdt = np.array(cmrf)
label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1) # Adjust to fit labels within the plot area
sns.heatmap(cmdt, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Decision Tree')
plt.show()

report = classification_report(true_labels, predicted_labels, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])


label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for Decision Tree')
plt.show()

for train_index, test_index in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    smote = SMOTE(random_state=2018)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    xgbc.fit(X_train, y_train)

    predicted_labels = np.append(predicted_labels, xgbc.predict(X_test))
    true_labels = np.append(true_labels, y_test)

accuracy = accuracy_score(true_labels, predicted_labels)
print("XG Boost Accuracy:", accuracy)
cmxg = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix XG Boost:")
print(cmxg)
cr = classification_report(true_labels, predicted_labels)
print("Classification Report XG Boost:")
print(cr)

cmxg = confusion_matrix(true_labels, predicted_labels)
cmxg = np.array(cmxg)
label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1) 
sns.heatmap(cmxg, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for XG Boost')
plt.show()


report = classification_report(true_labels, predicted_labels, output_dict=True)
data = {'precision':[], 'recall':[], 'f1-score':[], 'support':[]}
for key, value in report.items():
    if key in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    data['precision'].append(value['precision'])
    data['recall'].append(value['recall'])
    data['f1-score'].append(value['f1-score'])
    data['support'].append(value['support'])

label_names = ['Control', 'Depression', 'Schizophrenia']
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pd.DataFrame(data), annot=True, cmap='Blues', fmt='.2f',
            xticklabels=['Precision', 'Recall', 'F1-score', 'Support'],
            yticklabels=['Control', 'Depression', 'Schizophrenia'], ax=ax)
plt.title('Classification Report for XG Boost')
plt.show()