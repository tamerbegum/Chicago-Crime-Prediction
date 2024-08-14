import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix,roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
import xgboost as xgb
import shap
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df1_ = pd.read_csv('Chicago_Crimes_2001_to_2004.csv', on_bad_lines='skip')
df1 = df1_[df1_['Y Coordinate'] != '18 08:55:02 AM'].copy()
df1['Y Coordinate'] = df1['Y Coordinate'].astype(float)
df1['Latitude'] = df1['Latitude'].astype(float)
df2 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', on_bad_lines='skip')
df3 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', on_bad_lines='skip')
df4 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', on_bad_lines='skip')

merged_df = pd.concat([df1, df2, df3, df4])
df1, df2, df3, df4 = None, None, None, None  # For memory
merged_df.head()

# Remove duplicate
merged_df.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('Dataset shape after removing duplicate: ', merged_df.shape)
# Dataset shape after removing duplicate:  (6170812, 23)

# Remove uninformative information
col = ['Unnamed: 0', 'ID', 'Case Number', 'Updated On']
merged_df.drop(col, axis=1, inplace=True)

merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%m/%d/%Y %I:%M:%S %p')
merged_df.info()

print(merged_df['Arrest'].value_counts())

# Check missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1,
                           keys=['number of missing values', 'percentage of missing values'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(merged_df)

# random sampling
sampled_df = merged_df.sample(n=100000, random_state=42)
sampled_df.info()
missing_values_table(sampled_df)
sampled_df["Arrest"].value_counts()  # ratio still 2.5

# Split the sampled dataframe

indices = np.arange(len(sampled_df))
indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=0)
indices_train, indices_val = train_test_split(indices_train, test_size=0.2, random_state=0)

# Replace missing values:

def replace_missing_values_by_mode(df, var, train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mode = var_train.mode()[0]
    df[var] = df[var].fillna(var_train_mode)

def replace_missing_values_by_mean(df,var,train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mean = np.mean(var_train)
    df[var].fillna(var_train_mean,inplace = True)

variables_fill_mode = ['Community Area', 'Ward', 'Location', 'Location Description', 'District']

variables_fill_mean = ['X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude']

for variable in variables_fill_mode:
    replace_missing_values_by_mode(sampled_df, variable, indices_train)

for variable in variables_fill_mean:
    replace_missing_values_by_mean(sampled_df, variable, indices_train)


sampled_df_copy = sampled_df.copy()


# FEATURE SELECTION

def convert_categorical_to_numerical(df, columns):
    for col in columns:
        df[col] = pd.factorize(df[col])[0]
    return df.head()


non_numerical_columns = sampled_df_copy.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns.tolist()

convert_categorical_to_numerical(sampled_df_copy, non_numerical_columns)
sampled_df_copy.head()
sampled_df_copy['timestamp'] = pd.to_datetime(sampled_df_copy['Date']).astype('int64') / 10 ** 9  # Convert to seconds since epoch
sampled_df_copy['timestamp'].head()
print(sampled_df_copy.isnull().any())

Y_fs = sampled_df_copy['Arrest']
X_fs = sampled_df_copy.drop(['Date', 'Arrest'], axis=1)

# Train/val/test

indices_train_list = indices_train[0].tolist()
indices_test_list = indices_test[0].tolist()
indices_val_list = indices_val[0].tolist()

X_train = X_fs.iloc[indices_train,]
X_test = X_fs.iloc[indices_test,]
X_val = X_fs.iloc[indices_val,]

Y_train = Y_fs.iloc[indices_train,]
Y_test = Y_fs.iloc[indices_test,]
Y_val = Y_fs.iloc[indices_val,]

# XGBoost

# Fit model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
xgb_model.fit(X_train, Y_train)

# Predictions and accuracy on the validation set
y_val_labels_xgb = xgb_model.predict(X_val)
y_val_scores_xgb = xgb_model.predict_proba(X_val)[:, 1]
accuracy_val_xgb = accuracy_score(Y_val, y_val_labels_xgb)
print("Validation Accuracy: ", accuracy_val_xgb)

# AUC value on validation data
AUC_val_xgb = roc_auc_score(Y_val, y_val_scores_xgb)
print("Validation AUC: ", AUC_val_xgb)

# Tune hyperparameters with GridSearchCV

param_grid = {
    'max_depth': [1, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 800]
}
gridsearch = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc')
gridsearch.fit(X_train, Y_train)

# best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators':500}

# Retrain XGBoost model with best parameters
best_params = {'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 500}
best_xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, **best_params)
best_xgb_model.fit(X_train, np.ravel(Y_train))

# Predictions on the validation set
y_val_labels_xgb = best_xgb_model.predict(X_val)
y_val_scores_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

# Accuracy on validation data
accuracy_val_xgb = accuracy_score(Y_val, y_val_labels_xgb)
print("Validation Accuracy with Best Model: ", accuracy_val_xgb)

# AUC value on validation data
AUC_val_xgb = roc_auc_score(Y_val, y_val_scores_xgb)
print("Validation AUC with Best Model: ", AUC_val_xgb)

# Predictions on the test set
y_test_labels_xgb = best_xgb_model.predict(X_test)
y_test_scores_xgb = best_xgb_model.predict_proba(X_test)[:, 1]

# Accuracy on test data
accuracy_test_xgb = accuracy_score(Y_test, y_test_labels_xgb)
print("Test Accuracy with Best Model: ", accuracy_test_xgb)

# AUC value on test data
AUC_test_xgb = roc_auc_score(Y_test, y_test_scores_xgb)
print("Test AUC with Best Model: ", AUC_test_xgb)

# Confusion matrix for the test set
cf_xgb = confusion_matrix(Y_test, y_test_labels_xgb)
print("Confusion matrix of test set:\n", cf_xgb)

# ROC curve for test set
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_xgb)
roc_auc_xgb = AUC_test_xgb

plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC XGBoost test = %0.4f' % roc_auc_xgb)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # Plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


"""
CATBOOST
"""

# Fit model
catboost_model = CatBoostClassifier()
catboost_model.fit(X_train, Y_train)

# Predictions and accuracy on the validation set
y_val_labels_cat = catboost_model.predict(X_val)
y_val_scores_cat = catboost_model.predict_proba(X_val)[:, 1]
accuracy_val_cat = accuracy_score(Y_val, y_val_labels_cat)
print("Validation Accuracy: ", accuracy_val_cat)

# AUC value on validation data
AUC_val_cat = roc_auc_score(Y_val, y_val_scores_cat)
print("Validation AUC: ", AUC_val_cat)

# Hyperparameter Tuning
param_grid = {
    'depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
gridsearch = GridSearchCV(catboost_model, param_grid, cv=5, scoring='roc_auc')
gridsearch.fit(X_train, Y_train)

# Best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'depth': 5, 'learning_rate': 0.1, 'n_estimators':300}

# Retrain CatBoost model with best parameters
best_params = {'depth': 5, 'learning_rate': 0.1, 'n_estimators': 300}
best_catboost_model = CatBoostClassifier(**best_params,verbose=False)
best_catboost_model.fit(X_train, Y_train)

# Predictions on the validation set with the best model
y_val_labels_cat = best_catboost_model.predict(X_val)
y_val_scores_cat = best_catboost_model.predict_proba(X_val)[:, 1]

# Accuracy on validation data
accuracy_val_cat = accuracy_score(Y_val, y_val_labels_cat)
print("Validation Accuracy with Best Model: ", accuracy_val_cat)

# AUC value on validation data
AUC_val = roc_auc_score(Y_val, y_val_scores_cat)
print("Validation AUC with Best Model: ", AUC_val)

# Predictions on the test set
y_test_labels_cat = best_catboost_model.predict(X_test)
y_test_scores_cat = best_catboost_model.predict_proba(X_test)[:, 1]

# AUC value on test data
AUC_test_cat = roc_auc_score(Y_test, y_test_scores_cat)
print("Test AUC with Best Model: ", AUC_test_cat)

# Accuracy on test data
accuracy_test = accuracy_score(Y_test, y_test_labels_cat)
print("Test Accuracy with Best Model: ", accuracy_test)

# Confusion matrix for the test set
cf_cat = confusion_matrix(Y_test, y_test_labels_cat)
print("Confusion matrix of test set:\n", cf_cat)

# ROC curve for test set
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_cat)
roc_auc_cat = AUC_test_cat

plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC CatBoost test = %0.4f' % roc_auc_cat)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # Plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Feature importance: SHAP

explainer = shap.Explainer(best_catboost_model)
shap_values = explainer.shap_values(X_val)

# Summary plot
shap.summary_plot(shap_values, X_val)

# Plot the SHAP bar plot
shap.summary_plot(shap_values, X_val, plot_type="bar")
plt.show()

# Learning Curve

train_sizes, train_scores, test_scores = learning_curve(
    best_catboost_model, X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Accuracy")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation Accuracy")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc="best")
plt.grid()
plt.show()