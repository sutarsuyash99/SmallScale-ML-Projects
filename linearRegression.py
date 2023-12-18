# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Read the dataset and inspect the head to understand column names and existence
lin_reg_df = pd.read_csv('Real estate.csv')
lin_reg_df.head()

# Perform a basic check for null values to decide on imputation or drop
lin_reg_df.isnull().sum()

# Rename columns for easier usage
lin_reg_df.rename(columns={'No':'SL No', 'X1 transaction date':'Txn_Dt', 'X2 house age':'H_Age', 'X3 distance to the nearest MRT station':'Distance', 'X4 number of convenience stores':'Conv_stores', 'X5 latitude':'Lat', 'X6 longitude':'Long', 'Y house price of unit area':'Price_Area'}, inplace=True)

# Split dataset into target and feature variables
y = lin_reg_df['Price_Area']
X = lin_reg_df[['H_Age', 'Distance', 'Conv_stores']]

# Test the model with random_state values 0, 50, and 101, and report the best performance based on MAE, MSE, and RMSE
random_state_list = [0, 50, 101]
min_MAE, min_MSE, min_RMSE, best_rdm_st = float('inf'), float('inf'), float('inf'), 0

for rdm_st in random_state_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rdm_st)
    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)
    y_pred = model_LR.predict(X_test)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    
    if MAE < min_MAE:
        min_MAE = MAE
        min_MSE = MSE
        min_RMSE = RMSE
        best_rdm_st = rdm_st

    print("For random state = {}, the values are: ".format(rdm_st))
    print("Mean Absolute Error: ", MAE)
    print("Mean Squared Error: ", MSE)
    print("Root Mean Squared Error: ", RMSE)
    print("========================================================")
    print("\n")

# Report the random state that gave the best result and the respective values of MAE, MSE, and RMSE
best_st = best_rdm_st
print("Best Random State:", best_st)

best_MAE = min_MAE
print("Best Mean Absolute Error:", best_MAE)

best_MSE = min_MSE
print("Best Mean Squared Error:", best_MSE)

best_RMSE = min_RMSE
print("Best Root Mean Squared Error:", best_RMSE)

# Identify the most significant contributor to the LR model based on the coefficients
most_sig_wt, idx = 0, 0

for index, wt in enumerate(model_LR.coef_):
    if most_sig_wt < abs(wt):
        most_sig_wt = wt
        idx = index

most_sig_col = X.columns[idx]
print("Most Significant Contributor:", most_sig_col)

# Identify the intercept for the best model
intercept_val = round(model_LR.intercept_, 2)
print("Intercept Value:", intercept_val)
