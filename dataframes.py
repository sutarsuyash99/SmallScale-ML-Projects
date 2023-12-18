# Import necessary libraries
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Read the CSV file into a DataFrame
# %%
df = pd.read_csv("realestate.csv")

# Display the shape of the dataset
shape_data = df.shape
print(shape_data)

# Display information about the DataFrame
# %%
df.info
df.describe
print(df)

# Extract maximum values from specific columns
# %%
columnName1 = 'X2 house age'
columnName4 = 'X4 number of convenience stores'
X2_max = np.max(df[columnName1])
print(X2_max)
X4_max = np.max(df[columnName4])
print(X4_max)

# Check for null values in specific columns
# %%
columnName3 = 'X3 distance to the nearest MRT station'
columnName4 = 'X4 number of convenience stores'
columnNameY = 'Y house price of unit area'
X3_null = df[columnName3].isnull().sum()
X4_null = df[columnName4].isnull().sum()
Y_null = df[columnNameY].isnull().sum()
print(X3_null)
print(X4_null)

# Drop rows with null values and calculate mean of a column
# %%
df_drop = df.copy()
df_drop = df_drop.dropna(axis=0)
mean_X3_drop = np.mean(df_drop[columnName3])
print(mean_X3_drop)

# Fill null values with the median and calculate mean of a column
# %%
df_fill = df.copy()
df_fill = df_fill.fillna(df_fill.median())
mean_X3_fill = np.mean(df_fill[columnName3])
print(mean_X3_fill)

# Filter the DataFrame based on specific conditions
# %%
dataframe = df_fill.copy()
columnNameY = 'Y house price of unit area'
dataframe = dataframe.drop(dataframe[dataframe[columnNameY] > 80].index)
dataframe = dataframe.drop(dataframe[dataframe[columnName3] > 2800].index)
columnName6 = 'X6 longitude'
dataframe = dataframe.drop(dataframe[dataframe[columnName6] < 121.50].index)

# Convert a column using a conversion factor
# %%
conversionFactor = 91
dataframe['Y house price of unit area'] = dataframe['Y house price of unit area'].apply(lambda x: x * conversionFactor)

# Normalize a specific column
# %%
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(dataframe[[columnNameY]]), columns=[columnNameY])

# Calculate the mean of the normalized column
# %%
mean_norm_Y = normalized['Y house price of unit area'].mean()

# Print the mean of the normalized column
# %%
print("1.14 answer")
print(mean_norm_Y)

# Define a function to classify ages
# %%
def age_classifier(age):
    if age < 10:
        return 'New'
    elif age < 30:
        return 'Middle'
    else:
        return 'Old'

# Create a new column for age classification
# %%
df_age_classify = dataframe.copy()
df_age_classify['Age Class'] = df_age_classify['X2 house age'].apply(age_classifier)

# Count the occurrences of each age class
# %%
housesClassCount = df_age_classify['Age Class'].value_counts()
New_count = housesClassCount.get('New', 0)
Middle_count = housesClassCount.get('Middle', 0)
Old_count = housesClassCount.get('Old', 0)

# Set DataFrame index and find the row with the maximum value in a column
# %%
reindex_df = dataframe.set_index('No')
max_id = reindex_df[columnNameY].idxmax()

# Retrieve information from the row with the maximum value
# %%
expensiveHouse = reindex_df.loc[max_id]
txn_dt = expensiveHouse['X1 transaction date']
columnName2 = 'X2 house age'
house_age = expensiveHouse[columnName2]

# Filter DataFrame based on specific conditions
# %%
age_price_df = reindex_df.query("`X2 house age` <= 9.00 and `Y house price of unit area` > 27.00")

# Group by a column and calculate the mean
# %%
grouped_age_price_df = age_price_df.groupby('X4 number of convenience stores')['Y house price of unit area'].mean()
mean_val_conv_7 = round(grouped_age_price_df.get(7, 0), 2)
print(mean_val_conv_7)

# Plot a bar graph based on columns
# %%
import matplotlib.pyplot as plt

plot_graph = reindex_df.reset_index()
title = 'Mean House Price based on Convenience store proximity'
ylabel = 'Price of unit area'
xlabel = 'Number of convenience stores'
figsize = (6, 5)

plt.figure(figsize=figsize)
plt.bar(plot_graph['X4 number of convenience stores'], plot_graph['Y house price of unit area'])
plt.title(title)
plt.ylabel(ylabel)
plt.xlabel(xlabel)
plt.show()
