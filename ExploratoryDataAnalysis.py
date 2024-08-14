# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

"""Import datasets"""

#  Import datasets
df1_ = pd.read_csv('Chicago_Crimes_2001_to_2004.csv', on_bad_lines='skip')
df1_.head()
df1 = df1_[df1_['Y Coordinate'] != '18 08:55:02 AM'].copy()
df1['Y Coordinate'] = df1['Y Coordinate'].astype(float)
df1['Latitude'] = df1['Latitude'].astype(float)

# Import other datasets
df2 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', on_bad_lines='skip')
df3 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', on_bad_lines='skip')
df4 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', on_bad_lines='skip')

# Merge datasets
merged_df = pd.concat([df1, df2, df3, df4])
merged_df.head()

"""Data Understanding"""

# Check dataset information
def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(merged_df)


# Number of crimes per Primary Type
crime_count = pd.DataFrame(merged_df.groupby('Primary Type').size().sort_values(ascending=False))
print(crime_count)


# Check missing values
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['number of missing values', 'percentage of missing values'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(merged_df)


# Check types of variables
print(merged_df.columns)
print(merged_df.dtypes)

"""Pre-processing Data"""

# Remove duplicate
merged_df.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('Dataset shape after removing duplicate: ', merged_df.shape)

# Remove uninformative information
col = ['Unnamed: 0','ID','Case Number','Updated On']
merged_df.drop(col, axis=1, inplace=True)

# convert dates to pandas datetime format
merged_df['Updated On'] = pd.to_datetime(merged_df['Updated On'], format='%m/%d/%Y %I:%M:%S %p').dt.date

merged_df["Date"] = pd.to_datetime(merged_df["Date"], format='%m/%d/%Y %I:%M:%S %p')
merged_df["NEW_Date"] = merged_df["Date"].dt.date
merged_df["NEW_Time"] = merged_df["Date"].dt.hour
merged_df["Time_of_day"] = pd.cut(merged_df["NEW_Time"], bins=[-1, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"])

"""Replace missing values"""

# Check missing values
missing_values_table(merged_df)

# Replace missing values with mean
merged_df['X Coordinate'] = merged_df['X Coordinate'].fillna(merged_df['X Coordinate'].mean())
merged_df['Y Coordinate'] = merged_df['Y Coordinate'].fillna(merged_df['Y Coordinate'].mean())
merged_df['Latitude'] = merged_df['Latitude'].fillna(merged_df['Latitude'].mean())
merged_df['Longitude'] = merged_df['Longitude'].fillna(merged_df['Longitude'].mean())

# Replace missing values with mode
merged_df['Community Area'] = merged_df['Community Area'].fillna(merged_df['Community Area'].mode()[0])
merged_df['Ward'] = merged_df['Ward'].fillna(merged_df['Ward'].mode()[0])
merged_df['District'] = merged_df['District'].fillna(merged_df['District'].mode()[0])
merged_df['Location'] = merged_df['Location'].fillna(merged_df['Location'].mode()[0])
merged_df['Location Description'] = merged_df['Location Description'].fillna(merged_df['Location Description'].mode()[0])

# Check missing values again
merged_df.isnull().any().sum()

"""# Exploratory Data Analysis"""

# Number of crimes per district
plt.figure(figsize=(10, 6))
merged_df['District'].value_counts().plot(kind='bar')
plt.title('Number of crimes per district')
plt.xlabel('District')
plt.ylabel('Number of crimes')
plt.xticks(rotation=45)
plt.show()

# What are the most common crimes for each district?
# Group data by district and crime type, count occurrences, and reset index
district_crime_counts = merged_df.groupby(['District', 'Primary Type']).size().reset_index(name='Counts')

# Get the index of the maximum count for each district
idx = district_crime_counts.groupby(['District'])['Counts'].transform(max) == district_crime_counts['Counts']

# Filter the rows to only include the most common crimes for each district
most_common_crimes = district_crime_counts[idx]

# Display the result
print(most_common_crimes)

# First we check if the variable Arrest is or is not highly skewed by looking at the distribution of Arrest

arrest_counts = merged_df['Arrest'].value_counts()
arrest_percentage = arrest_counts / arrest_counts.sum() * 100

# Plotting
plt.figure(figsize=(8, 6))
ax = arrest_counts.plot(kind='bar', color=['lavender', 'lightblue'])

# Display percentage on each bar
for i, v in enumerate(arrest_counts):
    ax.text(i, v + 1, f'{arrest_percentage[i]:.1f}%', ha='center')

plt.title('Proportion of Crimes with and without Arrest')
plt.xlabel('Arrest Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Conclusion: Arrest is skewed. There are a lot more crimes reported without arrest.

# Correlation Matrix
# The values give the 'tendency' to increase (+) or decrease (-)
numeric_data = merged_df.select_dtypes(include='number')

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# The correlation coefficient has values between -1 to 1
# A value closer to 0 implies weaker correlation (exact 0 implying no correlation)
# A value closer to 1 implies stronger positive correlation
# A value closer to -1 implies stronger negative correlation

# Here we are looking at number of arrests per beat (more specific than district),
# we wanted to identify a threshold for which beats are most important
# in which we chose to do statistical analysis to determine the threshold (calculate the 90 percentile,
# so we focus on the top 10% of police beats with the highest number of arrests)

true_arrests_data = merged_df[merged_df['Arrest'] == True]

# Count the number of arrests in each beat
arrests_by_beat = true_arrests_data.groupby('Beat')['Arrest'].sum()
threshold = np.percentile(arrests_by_beat, 90)
# Only beats with more than 250 arrests
arrests_by_beat = arrests_by_beat[arrests_by_beat > threshold]

# Plotting
plt.figure(figsize=(12, 6))
arrests_by_beat.plot(kind='bar', color='skyblue')
plt.title('Number of Arrests by Police Beat (Arrest is True) - 90th percentile')
plt.xlabel('Police Beat')
plt.ylabel('Number of Arrests')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Which primary types occur the most when an arrest takes place (arrest=true)
arrested_data = merged_df[merged_df['Arrest'] == True]

arrested_primary_counts = arrested_data['Primary Type'].value_counts()
threshold = np.percentile(arrested_primary_counts, 75)
important_primary_types = arrested_primary_counts[arrested_primary_counts > threshold]

plt.figure(figsize=(10, 6))
important_primary_types.plot(kind='bar', color='lavender')
plt.title('Most Important Primary Types When Arrest is True')
plt.xlabel('Primary Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Slight tweak of the above script to only include the relevant primary types per district

# Filter the DataFrame for arrested incidents
arrested_data = merged_df[merged_df['Arrest'] == True]

# Group by District and Primary Type, and count how many times each Primary Type occurs in each District
district_primary_counts = arrested_data.groupby(['District', 'Primary Type']).size().unstack(fill_value=0)

# Calculate the threshold for important Primary Types
threshold = np.percentile(district_primary_counts.values.flatten(), 75)

# Plotting for each district
for district in district_primary_counts.index:
    # Get the counts of Primary Types for the current district
    primary_counts = district_primary_counts.loc[district]

    # Filter Primary Types based on the threshold for the current district
    important_primary_types = primary_counts[primary_counts > threshold]

    # Check if there are any Primary Types above the threshold;
    # districts that don't have any primary type counts above the threshold are neglected
    if not important_primary_types.empty:
        plt.figure(figsize=(8, 6))
        important_primary_types.plot(kind='bar', color=(0.5, 0.9, 0.7))
        plt.title(f'Count of Each Important Primary Type in District {district} Where Arrest is True')
        plt.xlabel('Primary Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# This could give you more information about the location
# When arrested, which crimes were caught the most per district. Also, about the distribution per district.
# Some districts are highly concentrated on a couple of primary types for example.
# That way, you can identify which districts are most 'responsible' for NARCOTICS, THEFT, BATTERY,
# which were the three most common primary types

# We saw that narcotics has the highest count, so we will look further into this

# Data where Arrest is True and Primary Type is Narcotics
filtered_data = merged_df[(merged_df['Arrest'] == True) & (merged_df['Primary Type'] == 'NARCOTICS')]
# Count how many times each Arrest description occurs
description_counts = filtered_data['Description'].value_counts()
threshold = np.percentile(arrested_primary_counts, 75)
important_description_counts = description_counts[description_counts > threshold]

# Plotting
plt.figure(figsize=(10, 6))
important_description_counts.plot(kind='bar', color='lightblue')
plt.title('Most Common Description for Narcotics-related Arrests')
plt.xlabel('Description')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# We know that the police arrests somebody, It is most likely Narcotics. Second place -> Battery, third -> Theft
# We are interested WHEN the police arrests someone (in general first),
# afterward specific for narcotics. (We can also dive deeper on Battery and Theft)
# This supports the previous analysis
arrested_data = merged_df[merged_df['Arrest'] == True]

arrested_time_of_day_counts = arrested_data['Time_of_day'].value_counts()

plt.figure(figsize=(10, 6))
arrested_time_of_day_counts.plot(kind='bar', color='lavender')
plt.title('Distribution of Time of the Day When Arrest is True')
plt.xlabel('Time of the Day')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Most are arrested in the evening/ afternoon

# Is this distribution more or less the same for Narcotics?

for primary_type in ['NARCOTICS','BATTERY','THEFT']:
  arrested_data = merged_df[(merged_df['Arrest'] == True) & (merged_df['Primary Type'] == primary_type)] # Filter or pivot
  arrested_time_of_day_counts = arrested_data['Time_of_day'].value_counts()
  plt.figure(figsize=(10, 6))
  arrested_time_of_day_counts.plot(kind='bar', color='lavender')
  plt.title('Distribution for {} of Time of the Day When Arrested'.format(primary_type))
  plt.xlabel('Time of the Day')
  plt.ylabel('Count')
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.show()

# Narcotics: same distribution as general
# Battery: Afternoon, evening, night
# Theft: primarily afternoon