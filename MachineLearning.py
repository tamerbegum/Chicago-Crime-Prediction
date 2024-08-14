import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, confusion_matrix
import tensorflow_decision_forests as tfdf
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from catboost import CatBoostClassifier
import shap
import tensorflow as tf
import xgboost as xgb

"""
Load the data that has been preprocessed
"""

file = 'Preprocessed_data.xlsx'
df_copy=pd.read_excel(file, index_col=0)

# List sheet names
sheet_names = pd.ExcelFile(file).sheet_names
print(sheet_names)

Arrest=pd.read_excel(file, sheet_name='Arrest', index_col=0)
indices_train=pd.read_excel(file, sheet_name='indices train', index_col=0)
indices_val=pd.read_excel(file, sheet_name='indices val', index_col=0)
indices_test=pd.read_excel(file, sheet_name='indices test', index_col=0)

indices_train_list = indices_train[0].tolist()
indices_test_list = indices_test[0].tolist()
indices_val_list = indices_val[0].tolist()

# Create train, validation, and test data

X_train = df_copy.iloc[indices_train_list]
X_test = df_copy.iloc[indices_test_list]
X_val = df_copy.iloc[indices_val_list]

Y_train = Arrest.iloc[indices_train_list]
Y_test = Arrest.iloc[indices_test_list]
Y_val = Arrest.iloc[indices_val_list]

X_train_total = pd.concat([X_train, X_val])  # Concatenate two data frames
Y_train_total = pd.concat([Y_train, Y_val])  # Concatenate two data frames


"""
KNN
"""

# Create a KNN classifier
knn = KNeighborsClassifier()

# Define the parameter grid to search
param_grid = {
    'n_neighbors': [100,200,300], # Try [100,1000,10000] -->
    # Best Parameters:  {'metric': 'euclidean', 'n_neighbors': 100, 'weights': 'uniform'} --> [100,200,300]
    'weights': ['uniform', 'distance'],  # Different weighting schemes
    'metric': ['euclidean', 'manhattan']  # Different distance metrics
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, Y_train.values.ravel())

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_) # Best Parameters:  {'metric': 'euclidean', 'n_neighbors': 100, 'weights': 'uniform'}

# KNN with optimal number of neighbors and weight set to inverse distance
clf_knn = KNeighborsClassifier(n_neighbors = 100, weights = 'uniform', metric = 'euclidean')
clf_knn.fit(X_train, Y_train.values.ravel())

# Make predictions on validation set
y_val_labels_knn = clf_knn.predict(X_val)
y_val_scores_knn = clf_knn.predict_proba(X_val)[:,1]

# Calculate the accuracy on validation data
accuracy_val_knn = accuracy_score(Y_val, y_val_labels_knn)
print("Validation Accuracy with Best Model: ", accuracy_val_knn)

# AUC value on validation data
AUC_val_knn = roc_auc_score(Y_val, y_val_scores_knn)
print("Validation AUC with Best Model: ", AUC_val_knn)

# Make predictions on test set
y_test_labels_knn = clf_knn.predict(X_test)
y_test_scores_knn = clf_knn.predict_proba(X_test)[:,1]

# calculate the accuracy on validation data
accuracy_test_knn = accuracy_score(Y_test, y_test_labels_knn)
print("Test Accuracy with Best Model: ",accuracy_test_knn)

# AUC value on test data
AUC_test_knn = roc_auc_score(Y_test, y_test_scores_knn)
print("Test AUC with Best Model: ",AUC_test_knn)

# Make confusion matrix for test set
cf_knn = confusion_matrix(Y_test, y_test_labels_knn)
print("Confusion matrix of test set:\n{}".format(cf_knn))

# Draw ROC curve
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_test_scores_knn)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic on test data') #title
plt.plot(fpr, tpr, 'b', label = 'AUC KNN test = %0.4f' %roc_auc) #label shown in the legend
plt.legend(loc = 'lower right') #plot a legend in the lower right
plt.plot([0, 1], [0, 1],'r--') #plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') #ylabel
plt.xlabel('False Positive Rate') #xlabel
plt.show()

# Validation Accuracy with Best Model:  0.87625
# Validation AUC with Best Model:  0.903931859778713
# Test Accuracy with Best Model:  0.8804
# Test AUC with Best Model:  0.8994891381772143
# Confusion matrix of test set:
# [[14014   453]
#  [ 1939  3594]]


"""
Decision tree
"""

clf_dt = tree.DecisionTreeClassifier()

min_samples_leaf_range = [100,200,300] # tried [100,1000,5000] --> Best model is: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 100, 'min_samples_split': 2}
#then try [100,200,300]
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15,20],
    'min_samples_split': [2, 5, 10,20,50],
    'min_samples_leaf': min_samples_leaf_range
}

gridsearch = GridSearchCV(clf_dt, param_grid, cv = 5)
gridsearch.fit(X_train , Y_train)

# Select best model
print ("Best model is: " + str(gridsearch.best_params_))
# Best model is: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 100, 'min_samples_split': 2}

# Given the optimal leaf size, rebuild/retrain tree.
clf_dt = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=100, max_depth = 10,  min_samples_split = 2)
clf_dt.fit(X_train, Y_train)

# Make predictions on validation set
y_val_labels_dt = clf_dt.predict(X_val)
y_val_scores_dt = clf_dt.predict_proba(X_val)[:,1]

# Calculate the accuracy on validation data
accuracy_val_dt = accuracy_score(Y_val, y_val_labels_dt)
print("Validation Accuracy with Best Model: ",accuracy_val_dt)

# AUC value on validation data
AUC_val_dt = roc_auc_score(Y_val, y_val_scores_dt)
print("Validation AUC with Best Model: ",AUC_val_dt)

# Make predictions on test set
y_test_labels_dt = clf_dt.predict(X_test)
y_test_scores_dt = clf_dt.predict_proba(X_test)[:,1]

# Calculate the accuracy on test data
accuracy_test_dt = accuracy_score(Y_test, y_test_labels_dt)
print("Test Accuracy with Best Model: ",accuracy_test_dt)

# AUC value on test data
AUC_test_dt = roc_auc_score(Y_test, y_test_scores_dt)
print("Test AUC with Best Model: ",AUC_test_dt)

# Make confusion matrix for test set
cf_dt = confusion_matrix(Y_test, y_test_labels_dt)
print("Confusion matrix of test set:\n{}".format(cf_dt))

# ROC curve
# calculate false positive and true positive rate
fpr, tpr, threshold = roc_curve (Y_test, y_test_scores_dt)
roc_auc = metrics.auc(fpr, tpr)

# Plot fpr and tpr
plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label = 'AUC DT test = %0.4f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--') # plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') # y- label
plt.xlabel('False Positive Rate') # x- label
plt.show()


# Validation Accuracy with Best Model:  0.8749375
# Validation AUC with Best Model:  0.901783444449124
# Test Accuracy with Best Model:  0.87755
# Test AUC with Best Model:  0.8996183140447986
# Confusion matrix of test set:
# [[14012   455]
#  [ 1994  3539]]

# Tree as pdf

dot_data = export_graphviz(
    clf_dt,
    out_file=None,
    feature_names=X_train.columns,
    class_names=['0', '1'],
    filled=True,
    rounded=True
)

with open("decision_tree.dot", "w") as dot_file:
    dot_file.write(dot_data)

import subprocess
subprocess.call(['dot', '-Tpdf', 'decision_tree.dot', '-o', 'decision_tree.pdf'])

"""
Random Forest
"""

rfc = RandomForestClassifier()

# Tuning
n_estimators = [100,500,1000]
param_grid = {'n_estimators': n_estimators}

gridsearch = GridSearchCV(rfc, param_grid, cv = 5)
gridsearch.fit(X_train , Y_train.values.ravel())

# Select best model
print("Best model is: " + str(gridsearch.best_params_)) #  {'n_estimators': 100}

# Given the optimal estimators, rebuild/retrain model.
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, Y_train.values.ravel())

# Make predictions on validation set
y_val_labels_rf = rfc.predict(X_val)
y_val_scores_rf = rfc.predict_proba(X_val)[:,1]

# Calculate the accuracy on validation data
accuracy_val_rf = accuracy_score(Y_val, y_val_labels_rf)
print("Validation Accuracy with Best Model: ",accuracy_val_rf)

# AUC value on validation data
AUC_val_rf = roc_auc_score(Y_val, y_val_scores_rf)
print("Validation AUC with Best Model: ",AUC_val_rf)

# Make predictions on test set
y_test_labels_rf = rfc.predict(X_test)
y_test_scores_rf = rfc.predict_proba(X_test)[:,1]

# Calculate the accuracy on test data
accuracy_test_rf = accuracy_score(Y_test, y_test_labels_rf)
print("Test Accuracy with Best Model: ",accuracy_test_rf)

# AUC value on test data
AUC_test_rf = roc_auc_score(Y_test, y_test_scores_rf)
print("Test AUC with Best Model: ",AUC_test_rf)

# Make confusion matrix for test set
cf_rf = confusion_matrix(Y_test, y_test_labels_rf)
print("Confusion matrix of test set:\n{}".format(cf_rf))

# ROC curve
# Calculate false positive and true positive rate
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_rf)
roc_auc = metrics.auc(fpr, tpr)

# Plot fpr and tpr
plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label = 'AUC RF test = %0.4f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--') # plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') # y- label
plt.xlabel('False Positive Rate') # x- label
plt.show()


# Validation Accuracy with Best Model:  0.819625
# Validation AUC with Best Model:  0.8594158279549011
# Test Accuracy with Best Model:  0.82555
# Test AUC with Best Model:  0.857612102134736
# Confusion matrix of test set:
# [[12782  1685]
#  [ 1804  3729]]


"""
Logistic Regression
"""

# Values for the hyperparameter/ regularization parameter C
Power=range(-12,10)
C=[]
for power in Power:
    C.append(2**power)

# Loop for hyperparameter tuning
# AUC values on the validation set are saved in the auc_val list
auc_val=[] # list initialization
for C_value in C:
    Model=LogisticRegression(C=C_value, solver='liblinear')
    Model.fit(X_train, np.ravel(Y_train))
    probs_val=Model.predict_proba(X_val)[:,1]
    auc_val.append(roc_auc_score(Y_val, probs_val))

print("Hyperparameter tuning ended")
index_maximal_auc=np.argmax(auc_val)
C_optimal=C[index_maximal_auc]

print("Optimal value for parameter C is %f" %C_optimal) # 0.007812

# Retrain model with best C-value
LR = LogisticRegression(C = 0.007812, solver='liblinear')
LR.fit(X_train, np.ravel(Y_train))

# Make predictions on validation set
y_val_labels_lr = LR.predict(X_val)
y_val_scores_lr = LR.predict_proba(X_val)[:,1]

# Calculate the accuracy on validation data
accuracy_val_lr = accuracy_score(Y_val, y_val_labels_lr)
print("Validation Accuracy with Best Model: ",accuracy_val_lr)

# AUC value on validation data
AUC_val_lr = roc_auc_score(Y_val, y_val_scores_lr)
print("Validation AUC with Best Model: ",AUC_val_lr)

# Compute the scores for the test instances
y_test_scores_lr = LR.predict_proba(X_test)[:,1]
y_test_labels_lr = LR.predict(X_test)

# Accuracy on test data
accuracy_test_lr = accuracy_score(Y_test, y_test_labels_lr)
print("Test Accuracy with Best Model: ",accuracy_test_lr)

# AUC value on test data
AUC_test_lr = roc_auc_score(Y_test, y_test_scores_lr)
print("Test AUC with Best Model: ",AUC_test_lr)

# Make confusion matrix for test set
cf_lr = confusion_matrix(Y_test, y_test_labels_lr)
print("Confusion matrix of test set:\n{}".format(cf_lr))

# ROC curve
fpr, tpr, threshold = metrics.roc_curve(Y_test, y_test_scores_lr)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic on test data') #title
plt.plot(fpr, tpr, 'b', label = 'AUC linear regression test = %0.4f' %roc_auc) #label shown in the legend
plt.legend(loc = 'lower right') #plot a legend in the lower right
plt.plot([0, 1], [0, 1],'r--') #plot diagonal indicating a random model
plt.ylabel('True Positive Rate') #ylabel
plt.xlabel('False Positive Rate') #xlabel
plt.show()

# Validation Accuracy with Best Model:  0.8580625
# Validation AUC with Best Model:  0.8786843807915741
# Test Accuracy with Best Model:  0.862
# Test AUC with Best Model:  0.8719865528171702
# Confusion matrix of test set:
# [[14171   296]
#  [ 2464  3069]]


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
plt.plot([0, 1], [0, 1], 'r--')
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


"""
ADABOOST
"""

# Fit model
adaboost_model = AdaBoostClassifier()
adaboost_model.fit(X_train, Y_train)

# Predictions and accuracy on the validation set
y_val_labels_ada = adaboost_model.predict(X_val)
y_val_scores_ada = adaboost_model.predict_proba(X_val)[:, 1]
accuracy_val_ada = accuracy_score(Y_val, y_val_labels_ada)
print("Validation Accuracy: ", accuracy_val_ada)

# AUC value on validation data
AUC_val_ada = roc_auc_score(Y_val, y_val_scores_ada)
print("Validation AUC: ", AUC_val_ada)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
gridsearch = GridSearchCV(adaboost_model, param_grid, cv=5, scoring='roc_auc')
gridsearch.fit(X_train, Y_train)

# Best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'learning_rate': 0.2, 'n_estimators':200}

# Retrain AdaBoost model with the best model parameters
best_params = {'learning_rate': 0.2, 'n_estimators': 200}
best_adaboost_model = AdaBoostClassifier(**best_params)
best_adaboost_model.fit(X_train, Y_train)

# Predictions on the validation set
y_val_labels_ada = best_adaboost_model.predict(X_val)
y_val_scores_ada = best_adaboost_model.predict_proba(X_val)[:, 1]

# Accuracy on validation data
accuracy_val_ada = accuracy_score(Y_val, y_val_labels_ada)
print("Validation Accuracy with Best Model: ", accuracy_val_ada)

# AUC value on validation data
AUC_val_ada = roc_auc_score(Y_val, y_val_scores_ada)
print("Validation AUC with Best Model: ", AUC_val_ada)

# Predictions on the test set
y_test_labels_ada = best_adaboost_model.predict(X_test)
y_test_scores_ada = best_adaboost_model.predict_proba(X_test)[:, 1]

# Accuracy on test data
accuracy_test_ada = accuracy_score(Y_test, y_test_labels_ada)
print("Test Accuracy with Best Model: ", accuracy_test_ada)

# AUC value on test data
AUC_test_ada = roc_auc_score(Y_test, y_test_scores_ada)
print("Test AUC with Best Model: ", AUC_test_ada)

# Confusion matrix for the test set
cf_ada = confusion_matrix(Y_test, y_test_labels_ada)
print("Confusion matrix of test set:\n", cf_ada)

# ROC curve for test set
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_ada)
roc_auc_ada = AUC_test_ada

plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC AdaBoost test = %0.4f' % roc_auc_ada)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # Plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


"""
XGBOOST
"""

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

# Best model parameters
print("Best model parameters: ", gridsearch.best_params_)
# Best model parameters:  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators':100}

# Retrain XGBoost model with best parameters
best_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
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
Neural Network
"""

# Create TensorFlow datasets combining features with labels
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    pd.concat([X_train_total, Y_train_total], axis=1), label='Arrest'
)
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    pd.concat([X_val, Y_val], axis=1), label='Arrest'
)

# Get input features
input_features = X_train.columns.tolist()

# Define the hyperparameters for tuning
hyperparams = {
    "num_trees": [100, 500, 1000],
    "max_depth": [3, 5, 7],
    "shrinkage": [0.1, 0.01],
}

# Iterate through hyperparameters
best_auc = 0.0
best_hparams = {}
best_val_auc = 0.0
best_val_accuracy = 0.0

for num_trees in hyperparams['num_trees']:
    for max_depth in hyperparams['max_depth']:
        for shrinkage in hyperparams['shrinkage']:
            # Train model with current hyperparameters
            model = tfdf.keras.GradientBoostedTreesModel(
                num_trees=num_trees,
                max_depth=max_depth,
                shrinkage=shrinkage,
                exclude_non_specified_features=False,
                features=[tfdf.keras.FeatureUsage(name=n) for n in input_features]
            )
            model.fit(train_ds, validation_data=val_ds, verbose=0)

            # Evaluate on validation set
            val_evaluation = model.evaluate(val_ds, return_dict=True)
            val_auc = val_evaluation.get('AUC', 0.0)
            val_accuracy = val_evaluation.get('Accuracy', 0.0)

            # Keep track of the best hyperparameters and validation scores
            if val_auc > best_val_auc:
                best_auc = val_auc
                best_hparams = {
                    "num_trees": num_trees,
                    "max_depth": max_depth,
                    "shrinkage": shrinkage,
                }
                best_val_auc = val_auc
                best_val_accuracy = val_accuracy

final_model = tfdf.keras.GradientBoostedTreesModel(
    exclude_non_specified_features=False,
    features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
    **best_hparams
)
final_model.fit(train_ds, validation_data=val_ds, verbose=0)

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    pd.concat([X_test, Y_test], axis=1), label='Arrest'
)

# Evaluate the final model on the test set
test_evaluation = final_model.evaluate(test_ds, return_dict=True)

# Compute AUC score for the test set
test_predictions = final_model.predict(test_ds)
test_labels = [label for _, label in test_ds.unbatch()]
test_auc = tf.keras.metrics.AUC()
test_auc.update_state(test_labels, test_predictions)
test_auc_score = test_auc.result().numpy()

# Compute accuracy for the test set
test_accuracy = tf.keras.metrics.BinaryAccuracy()
test_accuracy.update_state(test_labels, test_predictions)
test_accuracy_score = test_accuracy.result().numpy()
test_confusion_matrix = confusion_matrix(test_labels, test_predictions.round())

print("Test scores:")
print(f"Test AUC: {test_auc_score}")
print(f"Test Accuracy: {test_accuracy_score}")
print(f"Test Confusion Matrix:\n{test_confusion_matrix}")


# For validation:
# Define the neural network model
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the neural network model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# Train the neural network model
nn_history = nn_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=100, batch_size=32, verbose=0)

# Evaluate the neural network model on the validation set
y_val_scores = nn_model.predict(X_val)
y_val_labels = (y_val_scores > 0.5).astype(int)

# Accuracy on validation data
accuracy_val_nn = accuracy_score(Y_val, y_val_labels)
print("Validation Accuracy: ", accuracy_val_nn)

# AUC value on validation data
AUC_val_nn = roc_auc_score(Y_val, y_val_scores)
print("Validation AUC: ", AUC_val_nn)

# ROC curve for test set
# Compute AUC score for the test set
test_predictions = final_model.predict(test_ds)
test_labels = [label for _, label in test_ds.unbatch()]

# Calculate false positive and true positive rates
fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC Neural Network= {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic on test data')
plt.legend(loc="lower right")
plt.show()