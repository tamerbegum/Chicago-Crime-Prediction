import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

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
len(merged_df)

# Remove duplicate
merged_df.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('Dataset shape after removing duplicate: ', merged_df.shape)
# Dataset shape after removing duplicate:  (6170812, 23)

# Remove uninformative information
col = ['Unnamed: 0', 'ID', 'Case Number', 'Updated On']
merged_df.drop(col, axis=1, inplace=True)
len(merged_df)

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

# Random sampling:
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

sampled_df_copy.head()
sampled_df_copy['timestamp'] = pd.to_datetime(sampled_df_copy['Date']).astype('int64') / 10 ** 9  # Convert to seconds since epoch
sampled_df_copy['timestamp'].head()
print(sampled_df_copy.isnull().any())


# For FEATURE SELECTION:
# Convert categorical variables to numerical:
def convert_categorical_to_numerical(df, columns):
    for col in columns:
        df[col] = pd.factorize(df[col])[0]
    return df.head()


non_numerical_columns = sampled_df_copy.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns.tolist()

convert_categorical_to_numerical(sampled_df_copy, non_numerical_columns)

# Split Dataframe to target class and features

Y_fs = sampled_df_copy['Arrest']
X_fs = sampled_df_copy.drop(['Date', 'Arrest'], axis=1)
print(X_fs.columns)


# Use random forest for feature selection:
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

# Create a new Random Forest classifier with the best number of estimators
rfc = RandomForestClassifier(n_estimators=100)  # from tuning, it gave 1000

# Train the model with the best number of estimators
rfc.fit(X_train, Y_train)

# make predictions on validation set
y_val_labels_rf = rfc.predict(X_val)
y_val_scores_rf = rfc.predict_proba(X_val)[:, 1]

# calculate the accuracy on validation data
accuracy_val_rf = accuracy_score(Y_val, y_val_labels_rf)
print("Validation Accuracy with Best Model: ", accuracy_val_rf)  # 0.87625

# AUC value on validation data
AUC_val_rf = roc_auc_score(Y_val, y_val_scores_rf)
print("Validation AUC with Best Model: ", AUC_val_rf)  # 0.8964454530480231

# make predictions on test set
y_test_labels_rf = rfc.predict(X_test)
y_test_scores_rf = rfc.predict_proba(X_test)[:, 1]

# calculate the accuracy on test data
accuracy_test_rf = accuracy_score(Y_test, y_test_labels_rf)
print("Test Accuracy with Best Model: ", accuracy_test_rf)  # 0.87805

# AUC value on test data
AUC_test_rf = roc_auc_score(Y_test, y_test_scores_rf)
print("Test AUC with Best Model: ", AUC_test_rf)  # 0.891902854350674

# make confusion matrix for test set
cf_rf = confusion_matrix(Y_test, y_test_labels_rf)
print("Confusion matrix of test set:\n{}".format(cf_rf))

# [[ 3595  1938]
#  [  501 13966]]

# ROC curve
# calculate false positive and true positive rate
fpr, tpr, threshold = roc_curve(Y_test, y_test_scores_rf)
roc_auc = metrics.auc(fpr, tpr)


# plot fpr and tpr
plt.title('Receiver Operating Characteristic on test data')
plt.plot(fpr, tpr, 'b', label='AUC RF test = %0.4f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')  # plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')  # y- label
plt.xlabel('False Positive Rate')  # x- label
plt.show()

# Get feature importances
importances = rfc.feature_importances_

# Convert the importances into a more readable format
feature_importances = pd.DataFrame({'feature': X_fs.columns, 'importance': importances})

# Print the feature importances

feature_importances = feature_importances.sort_values('importance', ascending=False)
print(feature_importances)

import matplotlib.pyplot as plt

# Sort the feature importances by most important first
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Plot the importances
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

top_features = feature_importances[feature_importances['importance'] >= 0.045]['feature'].tolist()

# Print the top features
print("Top features with importance >= 0.045:")
print(top_features) 
# --> 'Primary Type', 'FBI Code', 'Description', 'IUCR', 'timestamp',
# 'Location', 'Location Description', 'Block', 'Arrest'

# #Using Pearson Correlation
plt.figure(figsize=(20,10))
cor = sampled_df_copy.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# After feature selection, we continue with the most important 9 features:

columns_to_keep = ['Primary Type', 'FBI Code', 'Description', 'IUCR', 'timestamp', 'Location', 'Location Description', 'Block', 'Arrest']

df = sampled_df_copy[columns_to_keep]

df.head()

sampled_df = None
sampled_df_copy = None

# BINNING:

# Frequency of Primary Type:

frequency = df["Primary Type"].value_counts()/len(df)
print(frequency)

# Calculate value counts and determine less frequent values:

value_counts = df["Primary Type"].value_counts()
total_records = len(df)
less_frequent_values = value_counts[value_counts / total_records < 0.03].index
df.loc[df['Primary Type'].isin(less_frequent_values), 'Primary Type'] = 'Others'

df['Primary Type'].value_counts()

# Frequency of Description:

frequency = df["Description"].value_counts()/len(df)
print(frequency)

# Calculate value counts and determine less frequent values
value_counts = df["Description"].value_counts()
total_records = len(df)
less_frequent_values = value_counts[value_counts / total_records < 0.04].index
df.loc[df['Description'].isin(less_frequent_values), 'Description'] = 'Others'

df['Description'].value_counts()

# Frequency of Block:

frequency = df["Block"].value_counts() / len(df)

print("Block frequencies:")
for block, freq in frequency.items():
    print(f"{block}: {freq:.5f}")

# Calculate value counts and determine less frequent values:

value_counts = df["Block"].value_counts()
total_records = len(df)
less_frequent_values = value_counts[value_counts / total_records < 0.0007].index
df.loc[df['Block'].isin(less_frequent_values), 'Block'] = 'Others'

df['Block'].value_counts()


# Standardize timestamp:

# Function to standardize a variable in a DataFrame
def standardize(df, var, train_ind):
    var_train = df[var].iloc[train_ind]
    var_train_mean = np.mean(var_train)
    var_train_std = np.std(var_train)
    df.loc[:, var] = (df[var] - var_train_mean) / var_train_std
    return df


# List of variables to standardize:
variables_to_standardize = ['timestamp']

# Apply standardization to the DataFrame
df = standardize(df, 'timestamp', indices_train)

df.head()

# Define the location mapping dictionary:

location_mapping = {
    'RESIDENCE': ['RESIDENCE', 'APARTMENT', 'RESIDENCE PORCH/HALLWAY', 'CHA APARTMENT', 'RESIDENCE-GARAGE', 'HOUSE',
                  'COACH HOUSE', 'ROOMING HOUSE'],
    'OTHER': ['OTHER', 'ALLEY', 'VACANT LOT/LAND', 'VACANT LOT', 'PARK PROPERTY', 'RAILROAD PROPERTY',
              'FOREST PRESERVE', 'WOODED AREA', 'JUNK YARD/GARBAGE DUMP', 'TRAILER', 'RIVER BANK'],
    'COMMERCIAL': ['GAS STATION', 'COMMERCIAL / BUSINESS OFFICE', 'SMALL RETAIL STORE', 'DEPARTMENT STORE',
                   'APPLIANCE STORE', 'HOTEL/MOTEL', 'MEDICAL/DENTAL OFFICE', 'SMALL RETAIL STORE', 'BARBERSHOP',
                   'NAIL SALON', 'DRUG STORE', 'CLEANING STORE', 'CONVENIENCE STORE', 'PAWN SHOP', 'POOLROOM',
                   'CLEANERS/LAUNDROMAT', 'NEWSSTAND', 'BARBER SHOP/BEAUTY SALON', 'GAS STATION DRIVE/PROP.',
                   'RESTAURANT', 'TAVERN/LIQUOR STORE', 'BAR OR TAVERN'],
    'STREET': ['STREET', 'SIDEWALK', 'GANGWAY', 'PARKING LOT', 'DRIVEWAY', 'DRIVEWAY - RESIDENTIAL', 'STAIRWELL',
               'HIGHWAY/EXPRESSWAY', 'EXPRESSWAY EMBANKMENT', 'CTA BUS', 'CTA PLATFORM', 'CTA TRAIN', 'CTA "L" TRAIN',
               'CTA PROPERTY', 'CTA BUS STOP', 'CTA "L" PLATFORM', 'CTA STATION', 'CTA TRACKS - RIGHT OF WAY',
               'TAXIWAY', 'LAGOON'],
    'COMMUNITY': ['CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'DAY CARE CENTER', 'COLLEGE/UNIVERSITY GROUNDS',
                  'SCHOOL, PUBLIC, BUILDING', 'HOSPITAL BUILDING/GROUNDS', 'SCHOOL, PRIVATE, GROUNDS',
                  'COLLEGE/UNIVERSITY RESIDENCE HALL', 'COUNTY JAIL', 'FIRE STATION', 'CHURCH PROPERTY',
                  'LIBRARY', 'PUBLIC GRAMMAR SCHOOL', 'SCHOOL YARD', 'PUBLIC HIGH SCHOOL', 'CHURCH', 'CHA PLAY LOT',
                  'CHA GROUNDS', 'YMCA', 'MOVIE HOUSE/THEATER', 'BOWLING ALLEY', 'COIN OPERATED MACHINE',
                  'SAVINGS AND LOAN', 'SEWER', 'LIVERY STAND OFFICE', 'GARAGE/AUTO REPAIR', 'CREDIT UNION',
                  'CHA BREEZEWAY', 'NEWSSTAND', 'BRIDGE', 'CHA LOBBY', 'PRAIRIE', 'FUNERAL PARLOR', 'OFFICE',
                  'TAXI CAB', 'TRUCKING TERMINAL', 'FACTORY', 'MOTEL', 'CONVENIENCE STORE', 'LAUNDRY ROOM',
                  'AIRPORT PARKING LOT', 'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA', 'LIVERY AUTO',
                  'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA', 'AIRPORT EXTERIOR - SECURE AREA',
                  'AIRPORT EXTERIOR - NON-SECURE AREA', 'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA',
                  'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',
                  'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'AIRPORT TRANSPORTATION SYSTEM (ATS)',
                  'NURSING HOME', 'VEHICLE - OTHER RIDE SERVICE', 'CLEANERS/LAUNDROMAT', 'EXPRESSWAY EMBANKMENT',
                  'GOVERNMENT BUILDING', 'POOLROOM', 'LAGOON']
}


# Create a function to map specific location descriptions to broader categories:
def map_location_to_category(location):
    for category, subcategories in location_mapping.items():
        if location in subcategories:
            return category
    return 'OTHER'


df['Location_Category'] = df['Location Description'].apply(map_location_to_category)

df['Location_Category'].unique()

# Location:

print(df['Location'].value_counts())
# most frequent ones
# (41.976290414, -87.905227221)
# (41.754592961, -87.741528537)
# (41.883500187, -87.627876698)

df[['Latitude', 'Longitude']] = df['Location'].str.extract(r'\((.*), (.*)\)').astype(float)

# Convert latitude and longitude columns to float
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)


print(df['Latitude'].unique())
print(df['Longitude'].unique())

# Choose the number of clusters (k)
k = 3

# Define the most frequent coordinates as initial cluster centers
initial_centers = [
    (41.976290414, -87.905227221),
    (41.754592961, -87.741528537),
    (41.883500187, -87.627876698)
]

# Fit k-means with initial cluster centers
kmeans = KMeans(n_clusters=k, init=np.array(initial_centers), n_init=1, max_iter=1)  # Use one iteration to directly assign labels
clusters = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# Update DataFrame with cluster labels
df['Cluster'] = clusters
df.head()


# Define the calculate_woe function
def calculate_woe(train_idx, test_idx, val_idx, feature, data, target):
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()
    val = data.iloc[val_idx].copy()

    total_positive = train[target].sum()
    total_negative = train[target].count() - total_positive

    woe_dict = {}
    for category in train[feature].unique():
        category_positive = train.loc[train[feature] == category, target].sum()
        category_negative = train.loc[train[feature] == category, target].count() - category_positive

        if category_positive == 0 and category_negative == 0:
            woe_dict[category] = 0
        elif category_positive == 0:
            woe_dict[category] = np.log(category_negative / total_negative)
        elif category_negative == 0:
            woe_dict[category] = -np.log(category_positive / total_positive)
        else:
            woe = np.log((category_negative / total_negative) / (category_positive / total_positive))
            woe_dict[category] = woe if not np.isinf(woe) and not np.isnan(woe) else 0

    data.loc[:, f'{feature}_WoE'] = data[feature].map(woe_dict)
    train.loc[:, f'{feature}_WoE'] = train[feature].map(woe_dict)
    test.loc[:, f'{feature}_WoE'] = test[feature].map(woe_dict)
    val.loc[:, f'{feature}_WoE'] = val[feature].map(woe_dict)

    return train, test, val


for feature in ['FBI Code', 'IUCR']:
    train_data, test_data, validation_data = calculate_woe(
        indices_train, indices_test, indices_val, feature, df, 'Arrest'
    )

df.head()

# Check for null values after applying the function:

null_columns = df.columns[df.isnull().any()]
null_counts = df[null_columns].isnull().sum()
print(null_counts)

# Calculate the mean value of the 'IUCR_WoE' column
average_iucr_woe = df['IUCR_WoE'].mean()
df['IUCR_WoE'] = df['IUCR_WoE'].fillna(average_iucr_woe)

print(df.isnull().any().sum())


# Dummy encoding:

def dummy_encode(df, column):
    dummy_df = pd.get_dummies(df[column], prefix=column, drop_first=True)
    dummy_df = dummy_df.astype(int)
    df = pd.concat([df, dummy_df], axis=1)
    return df


df = dummy_encode(df, 'Primary Type')
df = dummy_encode(df, 'Description')
df = dummy_encode(df, 'Block')
df = dummy_encode(df, 'Cluster')
df = dummy_encode(df, 'Location_Category')


df['Arrest'] = df['Arrest'].astype(int)

columns_to_drop = ['Location Description', 'Arrest', 'Location', 'Primary Type', 'Description', 'Block', 'Cluster', 'Location_Category', 'FBI Code', 'IUCR', 'Latitude', 'Longitude']

df_copy = df.drop(columns_to_drop, axis = 1)

Arrest = df['Arrest']


# Remove file if exists to update:

if os.path.exists('Preprocessed_data.xlsx'):
    os.remove('Preprocessed_data.xlsx')

print("Starting write process...")
with pd.ExcelWriter('Preprocessed_data.xlsx') as writer:
    df_copy.rename(columns=lambda x: x.replace(' ', '_')).to_excel(writer, sheet_name='Data_preprocessed')
    Arrest.to_excel(writer, sheet_name='Arrest')
    pd.DataFrame(indices_test).to_excel(writer, sheet_name='indices test')
    pd.DataFrame(indices_train).to_excel(writer, sheet_name='indices train')
    pd.DataFrame(indices_val).to_excel(writer, sheet_name='indices val')
print("Data saved to Preprocessed_data.xlsx")