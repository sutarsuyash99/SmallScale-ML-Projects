import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def main():
    df = pd.read_csv("realestate.csv")

    shape_data = df.shape
    print(shape_data)
    df.info
    df.describe
    print(df)
    columnName1 = 'X2 house age'
    columnName4 = 'X4 number of convenience stores'
    X2_max = np.max(df[columnName1])
    print(X2_max)
    X4_max = np.max(df[columnName4])
    print(X4_max)
    columnName3 = 'X3 distance to the nearest MRT station'
    columnName4 = 'X4 number of convenience stores'
    columnNameY = 'Y house price of unit area'
    X3_null = df[columnName3].isnull().sum()
    X4_null = df[columnName4].isnull().sum()
    Y_null = df[columnNameY].isnull().sum()
    print(X3_null)
    print(X4_null)

    df_drop = df.copy()
    df_drop = df_drop.dropna(axis = 0)
    mean_X3_drop = np.mean(df_drop[columnName3])
    print(mean_X3_drop)

    df_fill = df.copy()
    df_fill = df_fill.fillna(df_fill.median())
    mean_X3_fill = np.mean(df_fill[columnName3])
    print(mean_X3_fill)

    dataframe = df_fill.copy()
    columnNameY = 'Y house price of unit area'
    dataframe = dataframe.drop(dataframe[dataframe[columnNameY] > 80].index)
    dataframe = dataframe.drop(dataframe[dataframe[columnName3] > 2800].index)
    columnName6 = 'X6 longitude'
    dataframe = dataframe.drop(dataframe[dataframe[columnName6] < 121.50].index)

    conversionFactor = 10000*91
    dataframe['Y house price of unit area'] = dataframe['Y house price of unit area'].apply(lambda x: x/conversionFactor)

    scaler = MinMaxScaler()
    normalized = pd.DataFrame(scaler.fit_transform(dataframe[[columnNameY]]), columns=[columnNameY])

    print(type(normalized))
    normalized_df = normalized

    mean_norm_Y = normalized_df.mean()

    print(mean_norm_Y)

    def age_classifier(age):
        if age < 10:
            return 'New'
        elif age < 30:
            return 'Middle'
        else:
            return 'Old'

    df_age_classify = dataframe.copy()

    df_age_classify['Age Class'] = df_age_classify['X2 house age'].apply(age_classifier)

    housesClassCount = df_age_classify['Age Class'].value_counts()

    New_count = housesClassCount.get('New',0)
    Middle_count = housesClassCount.get('Middle',0)
    Old_count = housesClassCount.get('Old',0)

    reindex_df = dataframe.set_index('No')
    max_id = reindex_df[columnNameY].idxmax()

    expensiveHouse = reindex_df.loc[max_id]

    txn_dt = expensiveHouse[columnName1]
    columnName2 = 'X2 house age'
    house_age = expensiveHouse[columnName2]
    conv_st = expensiveHouse[columnName4]

    print(reindex_df)
    reindex_df['X2 house age'] = reindex_df['X2 house age'].astype(float)
    reindex_df['Y house price of unit area'] = reindex_df['Y house price of unit area'].astype(float)
    # print(reindex_df.dtypes)
    age_price_df = reindex_df.query('`X2 house age` <= 9.00 and `Y house price of unit area` > 27.00')
    grouped_age_price_df = reindex_df.groupby(columnName4)[columnNameY].mean()

    mean_val_conv_7 = reindex_df.iloc[7]
    mean_val_conv_7 = round(mean_val_conv_7, 2)

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






if __name__ == "__main__":
    main()