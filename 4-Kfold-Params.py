# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:20:04 2023

@author: Jfitz
"""
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, LeavePGroupsOut, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
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
#file = '4HrFeatures.csv'
#file='DayFeaturesk.csv'
#file='NightFeatures.csv'
file = '24HrFeaturesk.csv'
df = pd.read_csv(file)

# Prepare the data for modeling
X = df.drop(['id','class','date','category','counter','patientID'], axis=1)
y = df['class'].copy()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2018)

# Create a pipeline to impute missing values and scale the continuous variables
pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

# Fit the pipeline to the training data and transform the training and testing data
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

# Balance the target variable using SMOTE
smote = SMOTE(random_state=2018)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define the models and their hyperparameters
rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=2018)
dtc = DecisionTreeClassifier(max_depth=10, random_state=2018)
xgbc = xgb.XGBClassifier(n_estimators=200, max_depth=10, random_state=2018)
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=2018)
lgbmc = lgb.LGBMClassifier(n_estimators=200, max_depth=10, random_state=2018)

# Define the parameter grids for GridSearchCV
rfc_param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]}
dtc_param_grid = {'max_depth':[5, 10, 20]}
xgbc_param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]}
gbm_param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]}
lgbmc_param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]}

# Define the GridSearchCV objects with f1 as the scoring metric
rfc_grid = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, scoring='accuracy', cv=5)
dtc_grid = GridSearchCV(estimator=dtc, param_grid=dtc_param_grid, scoring='accuracy', cv=5)
xgbc_grid = GridSearchCV(estimator=xgbc, param_grid=xgbc_param_grid, scoring='accuracy', cv=5)
gbm_grid = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='accuracy', cv=5)
lgbmc_grid = GridSearchCV(estimator=lgbmc, param_grid=lgbmc_param_grid, scoring='accuracy', cv=5)

# Fit the GridSearchCV objects
rfc_grid.fit(X_train, y_train)
dtc_grid.fit(X_train, y_train)
xgbc_grid.fit(X_train, y_train)
gbm_grid.fit(X_train, y_train)
lgbmc_grid.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding scores
print("Best hyperparameters for Random Forest: ", rfc_grid.best_params_)
print("Best score for Random Forest: ", rfc_grid.best_score_)
print("Best hyperparameters for Decision Tree: ", dtc_grid.best_params_)
print("Best score for Decision Tree: ", dtc_grid.best_score_)
print("Best hyperparameters for XGBoost: ", xgbc_grid.best_params_)
print("Best score for XGBoost: ", xgbc_grid.best_score_)
print("Best hyperparameters for Gradient Boosting: ", gbm_grid.best_params_)
print("Best score for Gradient Boosting: ", gbm_grid.best_score_)
print("Best hyperparameters for LightGBM: ", lgbmc_grid.best_params_)
print("Best score for LightGBM: ", lgbmc_grid.best_score_)

#Train the models with the best hyperparameters on the full training set
rfc_best = RandomForestClassifier(**rfc_grid.best_params_, random_state=2018)
#rfc_best.fit(X_train, y_train)

dtc_best = DecisionTreeClassifier(**dtc_grid.best_params_, random_state=2018)
#dtc_best.fit(X_train, y_train)

xgbc_best = xgb.XGBClassifier(**xgbc_grid.best_params_, random_state=2018)
#xgbc_best.fit(X_train, y_train)

gbm_best = GradientBoostingClassifier(**gbm_grid.best_params_, random_state=2018)
#gbm_best.fit(X_train, y_train)

lgbmc_best = lgb.LGBMClassifier(**lgbmc_grid.best_params_, random_state=2018)
#lgbmc_best.fit(X_train, y_train)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2018)
# Create a pipeline to impute missing values and scale the continuous variables
pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler())

# Fit the pipeline to the training data and transform the training and testing data
X = pipeline.fit_transform(X)

# Detect and remove outliers using One-Class SVM
svm = OneClassSVM(nu=0.05)
outlier_mask = svm.fit_predict(X) == 1
X = X[outlier_mask]
y = y[outlier_mask]

# Balance the target variable using SMOTE
smote = SMOTE(random_state=2018)
X, y = smote.fit_resample(X, y)

# Create a pipeline to impute missing values and scale the continuous variables
pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())


n_splits = 10

# Perform k-fold cross-validation
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=2018)
# Perform stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2018)

models = {
    'Random Forest Classifier': RandomForestClassifier(**rfc_grid.best_params_, random_state=2018),
    'Decision Tree Classifier': DecisionTreeClassifier(**dtc_grid.best_params_, random_state=2018),
    'XGBoost Classifier': xgb.XGBClassifier(**xgbc_grid.best_params_, random_state=2018),
  #  'Gradient Boosting Classifier': GradientBoostingClassifier(**gbm_grid.best_params_, random_state=2018),
  #  'LightGBM Classifier': lgb.LGBMClassifier(**lgbmc_grid.best_params_, random_state=2018)
}
for model_name, model in models.items():
    print(f"Running {model_name}...")
    # lists to store metrics for each fold
    acc_scores = []
    auc_scores = []
    cm_list = []
    cr_list = []
    roc_auc_list = []  # list to store ROC AUC scores for each fold
    precision_list = []  # list to store precision scores for each fold
    f1_score_list = []  # list to store f1-score scores for each fold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the pipeline to the training data and transform the training and testing data
        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)

        # Fit the model
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Compute metrics
        acc_score = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Append metrics to lists
        acc_scores.append(acc_score)
        auc_scores.append(auc_score)
        cm_list.append(cm)
        cr_list.append(cr)
        roc_auc_list.append(auc_score)
        precision_list.append(precision)
        f1_score_list.append(f1)

    # calculate the average metrics over all folds
    avg_acc_score = np.mean(acc_scores)
    avg_auc_score = np.mean(auc_scores)
    avg_cm = np.mean(cm_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_f1_score = np.mean(f1_score_list)
    print(f"{model_name}")
    print("Average accuracy score over 10 folds : {:.2f}".format(avg_acc_score))
    print("Average ROC AUC score over 10 folds : {:.2f}".format(avg_auc_score))
    print("Average Confusion Matrix over 10 folds :\n{}".format(avg_cm))
    print("Average Precision over 10 folds : {:.2f}".format(avg_precision))
    print("Average F1-score over 10 folds : {:.2f}".format(avg_f1_score))

# =============================================================================
#         print(f"Fold: {fold}")    
#         print(f"Accuracy  ({model_name}): {acc_score}")
#         print(f"ROC AUC  ({model_name}): {auc_score}")
#         print(f"Confusion Matrix  ({model_name}):\n{cm}")
#         # Plot the confusion matrix as a heatmap
#         label_names = ['Control', 'Depression', 'Schizophrenia']
#         sns.set(font_scale=1) # Adjust to fit labels within the plot area
#         #sns.heatmap(confusion_mat, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g')
#         sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
#                xticklabels=label_names, yticklabels=label_names)
#         plt.xlabel('Predicted Labels')
#         plt.ylabel('True Labels')
#         plt.title('Confusion Matrix for  {model_name}')
#         plt.show()
#         print(f"Classification Report  ({model_name}):\n{cr}")
# =============================================================================
    



    # Plot the confusion matrix as a heatmap
    label_names = ['Control', 'Depression', 'Schizophrenia']
    sns.set(font_scale=1) # Adjust to fit labels within the plot area
    #sns.heatmap(confusion_mat, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g')
    sns.heatmap(avg_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g',
               xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
   # plt.title(f"Confusion Matrix for  {model_name}")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()
# =============================================================================
#     
#     # Plot ROC AUC curve
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []
#     aucs = []
#     fig, ax = plt.subplots()
#     for i, (train, test) in enumerate(skf.split(X, y)):
#         model.fit(X[train], y[train])
#         viz = plot_roc_curve(model, X[test], y[test], name='ROC fold {}'.format(i+1), alpha=0.3, lw=1, ax=ax)
#         interp_tpr = np.interp(mean_fpr, viz.fpr,  viz.tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(viz.roc_auc)
# =============================================================================
